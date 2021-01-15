import argparse
import glob
import os
import itertools

from model import *
import data_utils
import plot_utils

def parse_args():
    desc = "Pytorch implementation of Learning Invariant Representation for CL (IRCL) on the Split MNIST benchmark"
    parser = argparse.ArgumentParser(description=desc)

    # data
    parser.add_argument("--img_size", type=int, default=28, help="dimensionality of the input image")
    parser.add_argument("--channels", type=int, default=1, help="dimensionality of the input channels")
    parser.add_argument("--n_classes", type=int, default=10, help="total number of classes")

    # architecture
    parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent code")
    parser.add_argument('--n_hidden_cvae', type=int, default=300, help='number of hidden units in conditional variational autoencoder')
    parser.add_argument('--n_hidden_specific', type=int, default=20, help='number of hidden units in the specific module')
    parser.add_argument('--n_hidden_classifier', type=int, default=40, help='number of hidden units in the classification module')
    
    # training parameters
    parser.add_argument('--learn_rate', type=float, default=1e-2, help='learning rate for Adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=5, help='the number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='test Batch size')
    parser.add_argument("--log_interval", type=int, default=50, help="interval between logging")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--seed", type=int, default=1, help="seed") 

    # visualization
    parser.add_argument('--results_path', type=str, default='results',
                        help='the path of output images (generated and reconstructed)')
    parser.add_argument('--n_img_x', type=int, default=8,
                        help='number of images along x-axis')
    parser.add_argument('--n_img_y', type=int, default=8,
                        help='number of images along y-axis')


    return check_args(parser.parse_args())

def check_args(args):
    # results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)
    return args

def visualize(args, test_loader, encoder, decoder, epoch, n_classes, curr_task_labels):
    plotter = plot_utils.plot_samples(args.results_path, args.n_img_x, args.n_img_y, args.img_size, args.img_size)
    # plot samples of the reconstructed images from the first batch of the test set of the current task
    for test_batch_idx, (test_data, test_target) in enumerate(test_loader):                   
        x = test_data[0:plotter.n_total_imgs, :]
        x_id = test_target[0:plotter.n_total_imgs]
        x_id_onehot = get_categorical(x_id,n_classes)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z,_,_ = encoder(x)
            reconstructed_x = decoder(torch.cat([z, x_id_onehot], dim=1))
            reconstructed_x = reconstructed_x.reshape(plotter.n_total_imgs, args.img_size, args.img_size)
            plotter.save_images(x, name="/x_epoch_%02d" %(epoch) + ".jpg")
            plotter.save_images(reconstructed_x, name="/reconstructed_x_epoch_%02d" %(epoch) + ".jpg")
        break
    
    #plot pseudo random samples from the previous learned tasks
    z = Variable(Tensor(np.random.normal(0, 1, (plotter.n_total_imgs, args.latent_dim))))
    z_id = np.random.randint(0, curr_task_labels[-1]+1, size=[plotter.n_total_imgs])  
    z_id_one_hot = get_categorical(z_id, n_classes)
    decoder.eval()
    with torch.no_grad():
        pseudo_samples = decoder(torch.cat([z,Variable(Tensor(z_id_one_hot))],1))
        pseudo_samples = pseudo_samples.reshape(plotter.n_total_imgs, args.img_size, args.img_size)
        plotter.save_images(pseudo_samples, name="/pseudo_sample_epoch_%02d" % (epoch) + ".jpg")

def get_categorical(labels, n_classes=10):
    cat = np.array(labels.data.tolist())
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)

def generate_pseudo_samples(task_id, latent_dim, curr_task_labels, decoder, replay_count, n_classes=10):
    gen_count = sum(replay_count[0:task_id])
    z = Variable(Tensor(np.random.normal(0, 1, (gen_count, latent_dim))))    
    # this can be used if we want to replay different number of samples for each task
    for i in range(task_id):
        if i==0:
            x_id_ = np.random.randint(0, curr_task_labels[i][-1]+1, size=[replay_count[i]])  
        else:
            x_id_ = np.concatenate((x_id_,np.random.randint(curr_task_labels[i][0], curr_task_labels[i][-1]+1, size=[replay_count[i]])))

    np.random.shuffle(x_id_)
    x_id_one_hot = get_categorical(x_id_, n_classes)
    decoder.eval()
    with torch.no_grad():
        x = decoder(torch.cat([z,Variable(Tensor(x_id_one_hot))], 1))
    return x, x_id_

def evaluate(encoder, classifier, task_id, device, task_test_loader):
    correct_class = 0    
    n = 0
    classifier.eval()
    encoder.eval()
    with torch.no_grad():
        for data, target in task_test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            n += target.shape[0]
            z_representation,_,_ = encoder(data)
            model_output = classifier(data.view(data.shape[0], -1), z_representation)
            pred_class = model_output.argmax(dim=1, keepdim=True)
            correct_class += pred_class.eq(target.view_as(pred_class)).sum().item()

    print('Test evaluation of task_id: {} ACC: {}/{} ({:.3f}%)'.format(
         task_id, correct_class, n, 100*correct_class/float(n)))  

    return 100. * correct_class / float(n)

def train(args, optimizer_cvae, optimizer_C, encoder, decoder,classifer, train_loader, test_loader, curr_task_labels, task_id, device):
    ## loss ##
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    classification_loss = nn.CrossEntropyLoss()  
    encoder.train()
    decoder.train()
    classifer.train()
    for epoch in range(args.num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) 
            #---------------------------#
            ## train encoder-decoder ##
            #---------------------------#
            encoder.zero_grad()
            decoder.zero_grad()
            classifer.zero_grad()
            y_onehot = get_categorical(target, args.n_classes)
            encoded_imgs,z_mu,z_var = encoder(data)
            decoded_imgs = decoder(torch.cat([encoded_imgs, y_onehot], dim=1))
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)/args.batch_size
            rec_loss = pixelwise_loss(decoded_imgs, data)/args.batch_size
            cvae_loss = rec_loss + kl_loss
            cvae_loss.backward()
            optimizer_cvae.step()

            #---------------------------#
            ## train Classifer ##
            #---------------------------#
            encoder.zero_grad()
            decoder.zero_grad()
            classifer.zero_grad()
            z_representation,_,_ = encoder(data)
            # the classifier includes the specific module
            outputs = classifer(data.view(data.shape[0], -1), z_representation.detach())
            c_loss = classification_loss(outputs, target)
            c_loss.backward()
            optimizer_C.step()

            total_loss = cvae_loss.item() + c_loss.item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), total_loss)) 
                print("epoch %d: total_loss %03.2f cvae_loss %03.2f rec_loss %03.2f kl_loss %03.2f c_loss %03.2f" % (epoch, total_loss, cvae_loss.item(), rec_loss.item()/len(data), kl_loss.item(),c_loss.item()))
            
        if  epoch%2==0 or epoch+1 == args.num_epochs:
           test_acc = evaluate(encoder, classifer, task_id, device, test_loader)
           visualize(args, test_loader, encoder, decoder, epoch, args.n_classes, curr_task_labels)

    return test_acc

def main(args):
    print(args)
    # set seed
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("used device: " + str(device))
     
    ## DATA ##
    # Load data and construct the tasks 
    img_shape = (args.channels, args.img_size, args.img_size)
    task_labels = [[0,1],[2,3],[4,5],[6,7],[8,9]]
    num_tasks = len(task_labels)
    n_classes = args.n_classes
    num_replayed = [5000, 5000, 5000, 5000]
    train_dataset,test_dataset = data_utils.task_construction(task_labels)
    
    ## MODEL ##
    # Initialize encoder, decoder, specific, and classifier
    encoder = Encoder(img_shape, args.n_hidden_cvae, args.latent_dim)
    decoder = Decoder(img_shape, args.n_hidden_cvae, args.latent_dim, n_classes, use_label=True)
    classifier = Classifier(img_shape, args.latent_dim, args.n_hidden_specific, args.n_hidden_classifier, n_classes)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
        classifier.cuda()

    ## OPTIMIZERS ##
    optimizer_cvae = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.learn_rate)
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=args.learn_rate/50)

    test_loaders = []
    acc_of_task_t_at_time_t = [] # acc of each task at the end of learning it

    #------------------------------------------------------------------------------------------#
    #----- Train the sequence of CL tasks -----#
    #----------------------------------------------------------------------#
    for task_id in range(num_tasks):
        print("Strat training task#" + str(task_id))
        if task_id>0:
            # generate pseudo-samples of previous tasks
            gen_x,gen_y = generate_pseudo_samples(task_id, args.latent_dim, task_labels, decoder, num_replayed)
            gen_x = gen_x.reshape([gen_x.shape[0],img_shape[1],img_shape[2]])
            train_dataset[task_id-1].data = (gen_x*255).type(torch.uint8)
            train_dataset[task_id-1].targets = Variable(Tensor(gen_y)).type(torch.long)
            # concatenate the pseduo samples of previous tasks with the data of the current task
            train_dataset[task_id].data = torch.cat((train_dataset[task_id].data,train_dataset[task_id-1].data))
            train_dataset[task_id].targets =  torch.cat((train_dataset[task_id].targets, train_dataset[task_id-1].targets))

        train_loader = data_utils.get_train_loader(train_dataset[task_id], args.batch_size)
        test_loader = data_utils.get_test_loader(test_dataset[task_id], args.test_batch_size)
        test_loaders.append(test_loader)
        # train current task
        test_acc = train(args, optimizer_cvae, optimizer_C, encoder, decoder, classifier, train_loader, test_loader, task_labels[task_id], task_id, device)
        acc_of_task_t_at_time_t.append(test_acc)
        print('\n')
    #------------------------------------------------------------------------------------------#
    #----- Performance on each task after training the whole sequence -----#
    #----------------------------------------------------------------------#
    ACC = 0
    BWT = 0
    for task_id in range(num_tasks):
        task_acc = evaluate(encoder, classifier, task_id, device, test_loaders[task_id])
        ACC += task_acc
        BWT += (task_acc - acc_of_task_t_at_time_t[task_id])
    ACC = ACC/len(task_labels)
    BWT = BWT/len(task_labels)-1
    print('Average accuracy in task agnostic inference (ACC):  {:.3f}'.format(ACC))
    print('Average backward transfer (BWT): {:.3f}'.format(BWT))


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    # main
    main(args)
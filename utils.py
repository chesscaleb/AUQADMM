#utils

#MNIST DataLoader
def mnist_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size
    trainLoader = torch.utils.data.DataLoader(
        datasets.MNIST(root = './',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=train_batch_size,
        shuffle=True)
    testLoader = torch.utils.data.DataLoader(
        datasets.MNIST(root = './',
                   train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=test_batch_size,
        shuffle=False)
    return trainLoader, testLoader


#CIFAR10 DataLoader
def cifar_loaders(train_batch_size, test_batch_size=None, augment=False):
    if test_batch_size is None:
        test_batch_size = train_batch_size
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
    if augment:
        transforms_list = [transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, 4),
                            transforms.ToTensor(),
                            normalize]
    else:
        transforms_list = [transforms.ToTensor(),
                           normalize]
    train_dset = datasets.CIFAR10('data',
                              train=True,
                              download=True,
                              transform=transforms.Compose(transforms_list))
    test_dset = datasets.CIFAR10('data',
                             train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                             ]))
    trainLoader = torch.utils.data.DataLoader(train_dset, batch_size=train_batch_size,
                                              shuffle=True, pin_memory=True)
    testLoader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size,
                                             shuffle=False, pin_memory=True)
    return trainLoader, testLoader


#SVHN DataLoader
def svhn_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size
    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                      std=[0.1980, 0.2010, 0.1970])
    train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root='data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ]),
            ),
            batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root='data', split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def Initialization(SAMPLE_NUM_EACH_WORKER, DATASET_NAME, LOSS_NAME, n=None, num_workers=10):
    
    train_batch_size = 1
    if DATASET_NAME == 'MNIST':
        [trainLoader, testLoader] = mnist_loaders(train_batch_size, test_batch_size=None)
    elif DATASET_NAME == 'CIFAR10':
        [trainLoader, testLoader] = cifar_loaders(train_batch_size, test_batch_size=None, augment=False)
    elif DATASET_NAME == 'SVHN':
        [trainLoader, testLoader] = svhn_loaders(train_batch_size, test_batch_size=None)
    
    trainsets = Generate_and_Classify_Trainsets(SAMPLE_NUM_EACH_WORKER, DATASET_NAME, trainLoader, LOSS_NAME)

    if DATASET_NAME == 'MNIST':
        m = 784
    elif DATASET_NAME == 'CIFAR10':
        m = 3072
    elif DATASET_NAME == 'SVHN':
        m = 3072
        

    params = []

    if LOSS_NAME == 'Multinomial':
        for i in range(10):
            params.append(torch.randn(m,10,requires_grad=True))
        return [params, trainsets]
    elif LOSS_NAME == 'Elastic_Net':
        for i in range(10):
            params.append(torch.randn(m,1,requires_grad=True))
        return [params, trainsets]
    elif LOSS_NAME == 'Smoothed_SVM':
        for i in range(2):
            params.append(torch.randn(m,1,requires_grad=True))
        return [params, trainsets]
    else:
        for i in range(num_workers):
            params.append(torch.randn(m,n,requires_grad=True))
        return [params, trainsets]


def Generate_and_Classify_Trainsets(Number_of_Samples_each_worker, Target_Dataset_Name, TrainLoader, LOSS_NAME):
    #Number_of_Samples_each_worker: number of samples assigned to each worker, like 2000, 2500 or so
    #Target_Dataset_Name: 'MNIST' or 'CIFAR10' or 'SVHN'
    #TrainLoader: trainloader containing datasets
    #LOSS_NAME: 'Multinomial' or 'Elastic_Net' or 'Smoothed_SVM'
    N = Number_of_Samples_each_worker
    if Target_Dataset_Name == 'MNIST':
        M = 28*28
    elif Target_Dataset_Name == 'CIFAR10' or Target_Dataset_Name == 'SVHN':
        M = 3072

    if LOSS_NAME != 'Smoothed_SVM':
        Zeros = []; Ones = []; Twos = []; Threes = []; Fours = []; Fives = []; Sixs = []; Sevens = []; Eights = []; Nines = []

        for data in TrainLoader:
            D, L = data
            if L[0] == 0:
                Zeros.append(D)
            elif L[0] == 1:
                Ones.append(D)
            elif L[0] == 2:
                Twos.append(D)
            elif L[0] == 3:
                Threes.append(D)
            elif L[0] == 4:
                Fours.append(D)
            elif L[0] == 5:
                Fives.append(D)
            elif L[0] == 6:
                Sixs.append(D)
            elif L[0] == 7:
                Sevens.append(D)
            elif L[0] == 8:
                Eights.append(D)
            elif L[0] == 9:
                Nines.append(D)

        X0 = Zeros[0].view(-1,M)
        X1 = Ones[0].view(-1,M)
        X2 = Twos[0].view(-1,M)
        X3 = Threes[0].view(-1,M)
        X4 = Fours[0].view(-1,M)
        X5 = Fives[0].view(-1,M)
        X6 = Sixs[0].view(-1,M)
        X7 = Sevens[0].view(-1,M)
        X8 = Eights[0].view(-1,M)
        X9 = Nines[0].view(-1,M)
        L0 = torch.zeros(N, dtype=torch.long)
        L1 = torch.ones(N, dtype=torch.long)
        L2 = 2*torch.ones(N, dtype=torch.long)
        L3 = 3*torch.ones(N, dtype=torch.long)
        L4 = 4*torch.ones(N, dtype=torch.long)
        L5 = 5*torch.ones(N, dtype=torch.long)
        L6 = 6*torch.ones(N, dtype=torch.long)
        L7 = 7*torch.ones(N, dtype=torch.long)
        L8 = 8*torch.ones(N, dtype=torch.long)
        L9 = 9*torch.ones(N, dtype=torch.long)

        for i in range(N-1):
            A = Zeros[i+1].view(-1,M)
            X0 = torch.cat((X0,A),0)
            A = Ones[i+1].view(-1,M)
            X1 = torch.cat((X1,A),0)
            A = Twos[i+1].view(-1,M)
            X2 = torch.cat((X2,A),0)
            A = Threes[i+1].view(-1,M)
            X3 = torch.cat((X3,A),0)
            A = Fours[i+1].view(-1,M)
            X4 = torch.cat((X4,A),0)
            A = Fives[i+1].view(-1,M)
            X5 = torch.cat((X5,A),0)
            A = Sixs[i+1].view(-1,M)
            X6 = torch.cat((X6,A),0)
            A = Sevens[i+1].view(-1,M)
            X7 = torch.cat((X7,A),0)
            A = Eights[i+1].view(-1,M)
            X8 = torch.cat((X8,A),0)
            A = Nines[i+1].view(-1,M)
            X9 = torch.cat((X9,A),0)

        trainset1 = [X0,L0]
        trainset2 = [X1,L1]
        trainset3 = [X2,L2]
        trainset4 = [X3,L3]
        trainset5 = [X4,L4]
        trainset6 = [X5,L5]
        trainset7 = [X6,L6]
        trainset8 = [X7,L7]
        trainset9 = [X8,L8]
        trainset10 = [X9,L9]

        return [trainset1, trainset2, trainset3, trainset4, trainset5, trainset6, trainset7, trainset8, trainset9, trainset10]

    else:
        Zeros = []; Ones = []

        for data in trainLoader_min:
            D, L = data
            if L[0] == 0:
                Zeros.append(D)
            elif L[0] == 1:
                Ones.append(D)

        Classified_MINIST = {}
        Classified_MINIST['0'] = Zeros
        Classified_MINIST['1'] = Ones

        X0 = Zeros[0].view(-1,M)
        X1 = Ones[0].view(-1,M)

        L0 = -1*torch.ones(N,1, dtype=torch.long)
        L1 = torch.ones(N,1, dtype=torch.long)

        for i in range(N-1):
            A = Zeros[i+1].view(-1,M)
            X0 = torch.cat((X0,A),0)
            A = Ones[i+1].view(-1,M)
            X1 = torch.cat((X1,A),0)

        trainset1 = [X0,L0]
        trainset2 = [X1,L1]

        return [trainset1, trainset2]


#Lanczos Algorithm
def manual_Lanczos(f, x, q1, rank):
    ##INPUTS:
    #f: objective function
    #x: the point at which the gradient and Hessian are evaluated
    #q1: initial vector
    #rank: desired rank number for the approximation QTQ^t
    
    ##OUTPUTS:
    #Qt: transpose of Q
    #T: the triadiagonal matrix T
    
    k = 0
    beta = 1.0
    q = torch.zeros_like(q1)
    r = q1
    dim1 = q1.shape[0]
    dim2 = q1.shape[1]
    x.requires_grad = True

    Qt = []
    a = []
    b = []
    
    
    while k < rank:
        q_prev = q.clone().detach()
        q = r/beta
        k = k+1
        
        #Calculate A*q
        y = f(x)
        dydx = torch.autograd.grad(outputs=f(x), inputs=x,
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
        Aq = torch.autograd.grad(outputs=dydx, inputs=x, grad_outputs=q)[0]
        
        alpha = (q * Aq).sum()
        r = Aq - alpha*q - beta*q_prev
        beta = torch.norm(r)
        
        Qt.append(torch.transpose(q,0,1).reshape(dim1*dim2).tolist())
        a.append(alpha.item())
        if k < rank:
            b.append(beta.item())
            
    a = torch.tensor(a); b = torch.tensor(b)
    T = torch.diag(a) + torch.diag(b,1) + torch.diag(b,-1)
    
    return [torch.tensor(Qt).clone().detach(), T.clone().detach()]

def FullLoss(loss_function, trainset, x, N, M):
    return loss_function(trainset, x, N, M)

def Multinomial(trainset, x, N, M):
    loss = nn.CrossEntropyLoss()
    output = 0
    X, y = trainset
    X = X.view(-1, M)
    input = torch.mm(X, x)
    output += loss(input,y)/(N*1.0)
    return output

def ElasticNet(trainset, x, N, M): 
    loss = nn.MSELoss()
    output = 0
    X, y = trainset
    X = X.view(-1, M)
    input = torch.mm(X, x)
    y = y.reshape(X.shape[0],1)
    y = y.type(torch.FloatTensor)
    output += loss(input,y)
    return output

def SmoothedSVM(trainset, x, N, M): 
    def j(u, eps):
        return 0.5*(u+torch.sqrt(eps**2+u**2))

    def hinge_loss(X, u, y, eps):
        return torch.mean(j(1-y*torch.mm(X,u),eps))

    X, y = trainset
    eps = 1.0/5000.0
    return hinge_loss(X, x, y, eps)

#AUQADMM CLASS

class AUQADMM:
    def __init__(self, params, regularizer, rho1, rho2, trainsets, LOSS_NAME):
        self.LOSS_NAME = LOSS_NAME
        self.U = params
        dim1 = params[0].shape[0]; dim2 = params[0].shape[1];
        self.dim = 1.0*dim1*dim2
        self.Workers = len(self.U)
        self.regularizer = regularizer
        self.rho1 = rho1
        self.rho2 = rho2
        self.trainsets = trainsets

        self.iteration_number = []
        self.objfunc = []
        self.lambdas = []
        
        self.dlambdas = []
        self.dus = []

        self.Hessians = []
        self.diagonalweights = []

        self.V = torch.zeros_like(self.U[0])
        self.M = [*self.V.shape][0]

        self.loss = []

        for u in self.U:
            self.lambdas.append(torch.zeros_like(u))
            self.dus.append(torch.zeros_like(u))
            self.dlambdas.append(torch.zeros_like(u))
            self.Hessians.append(0)
            self.diagonalweights.append(0)

    #Affine Normalization such that all elements of target X are in a range [a,b]
    def affine_normalize(self, a, b, X):
        #a,b: range of cutoff
        #X: target

        q = max(X).item()
        p = min(X).item()
        m = (b-a)/(q-p)
        n = a - m*p
        return m*X+n

    #Generate Diagonal Weights, return weight and the normalization factor gamma
    def GenerateDiagWeights(self, trainset, rank, u, a, b, N, M, LOSS_NAME):
        dim1 = u.shape[0]
        dim2 = u.shape[1]
        
        def f(c):
            return FullLoss(trainset, c, N, M, LOSS_NAME)

        q1 = torch.randn(dim1,dim2)
        q1 = 1.0/torch.norm(q1)*q1
        [Qt, T] = manual_Lanczos(f, u, q1, 10)
        Hessian = torch.mm(torch.transpose(Qt,0,1),torch.mm(T,Qt))

        Hdiag = torch.diagonal(Hessian, 0)
        Hdiag = Hdiag.view(-1)

        Hdiag = self.affine_normalize(a, b, Hdiag) #Normalize the Hessian such that all elements are in [a, b]
        Hdiag = Hdiag.view(dim1, dim2)

        return Hdiag
    
    #LBFGS Algorithm for U Updates
    def UpdateLocalLBFGS(self, U, V, lam, w, trainset, epochs, N, M, LOSS_NAME):
        optimizer = optim.LBFGS([U],max_iter=4, history_size=20 ,lr = 1e-3)

        for epoch in range(epochs):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                loss = self.LocalCost(trainset, U, V, lam, w, N, M, LOSS_NAME)
                if loss.requires_grad:
                    loss.backward(retain_graph=True)
                return loss
            optimizer.step(closure)
        return U
    
    #Lambda Updates
    def UpdateLambda(self, U, V, lambdas, w):
        with torch.no_grad():    
            for count, u in enumerate(U):
                lambdas[count] = copy.deepcopy(lambdas[count])
                lambdas[count] = torch.add(lambdas[count], w[count]*(V - u))
        return lambdas

    #Interval Update
    def UpdateInterval(self, k, a1, b1, a, b, K):
        if k%K==0:
            i = k*1.0/(K*1.0)
            gam = 1.0/(i+1)**2*b1/a1+1-1/(i+1)**2
            b = gam*a
        return [a,b]
    
    #Local Cost for workers u_j in ADMM
    def LocalCost(self, trainset, u, v, lam, w, N, M, LOSS_NAME):
        objfunc = FullLoss(trainset, u, N, M, LOSS_NAME)
        extra_terms = 0
        diff = v - u
        res = diff + 1/w*lam
        
        extra_terms = 0.5*((res*w*res).sum())

        return objfunc + extra_terms    

    #Regularizer Loss in ADMM
    def regularizer_loss(self, x_list, y_list, z, tau_list):
        extra_terms = 0
        for i in range(len(x_list)):
            norm_sq = (z - x_list[i] + y_list[i]/tau_list[i]).pow(2).sum()
            extra_terms += 0.5*tau_list[i]*norm_sq
        return self.regularizer(z) + extra_terms

    #Proximal Algorithm for V Update
    def proximal(self, rho1, rho2, U, lambdas, diag_Weight_list):
        threshold = torch.nn.Threshold(0,0)
        X = torch.zeros_like(U[0])
        K = rho2*torch.ones_like(diag_Weight_list[0])
        for count, u in enumerate(U):
            X = X + diag_Weight_list[count].clone()*u.clone()-lambdas[count]
            K = K + diag_Weight_list[count]
        value = rho1
        V = 1.0/K*threshold(abs(X)-value)*torch.sign(X)
        return V

    #Optimization Function
    def fit(self, maxiter=1, X_EPOCHS=3, abs_tol=1e-5, rel_tol=1e-4, rank=5, a=0.5, b=1.5, closure=None, K=1000):
        a0 = a; b0 = b
        primal_residual = [] #primal_residual
        dual_residual = [] #dual_residual
        M = self.M
        LOSS_NAME = self.LOSS_NAME
        
        
        for i in range(maxiter):
            t = time.time()

            V_previous = self.V.clone().detach()
                
            #Generate weights and update u
            for it, trainset in enumerate(self.trainsets):
                #Generate Diagonal Hessian Weights
                self.diagonalweights[it] = self.GenerateDiagWeights(trainset, rank, self.U[it].clone().detach(), a, b, 1.0, M, LOSS_NAME) ###
                
                #Update U
                self.U[it] = self.UpdateLocalLBFGS(self.U[it], self.V, self.lambdas[it], self.diagonalweights[it], trainset, X_EPOCHS, 1.0, M, LOSS_NAME) ###
            
            #Update Interval
            a, b = self.UpdateInterval(i+1, a0, b0, a, b, K)
            print('a', a)
            print('b', b)

            #Save the norm of u and total loss in each iteration
            u_norm = []
            for u in self.U:
                u_norm.append(torch.norm(u))

            #Update v and lambda
            with torch.no_grad():
                self.V = self.proximal(self.rho1, self.rho2, self.U, self.lambdas, self.diagonalweights)
                self.lambdas = self.UpdateLambda(self.U, self.V, self.lambdas, self.diagonalweights)
            
            #Save the norm of lambda
            lam_norm = []
            for lam in self.lambdas:
                lam_norm.append(torch.norm(lam))
            
            loss = 0
            for it, trainset in enumerate(self.trainsets):
                loss += FullLoss(trainset, self.V, self.Workers, M, LOSS_NAME)    
            loss += self.regularizer(self.V)
            self.loss.append(loss)

            #Only for printing and formatting purpose
            info_list = [(i+1), loss]
            C_cg = (b/a-1.0)*maxiter**2+1
            if i == 0:
                print()
                print('Convergence Constant: ', C_cg)
                print()
                print('%s    %s'
                % ('k^th',  'f+g'))
            for index in range(len(info_list)):
                if index == 0:
                    print('%d'%info_list[index], end='       ')
                else:
                    print('%.2e'%info_list[index], end='       ')
            print()

            #Save previous weights
            previous_weights = copy.deepcopy(self.diagonalweights)

            with torch.no_grad():
                #Stopping Criteria
                pr = 0
                dr = 0
                u_norm_sq = 0
                v_norm_sq = (self.V*self.V).sum().item()
                lam_norm_sq = 0
                
                for count, u in enumerate(self.U):
                    pr += ((self.V-u)*(self.V-u)).sum().item()
                    vec = V_previous - self.V
                    dr += (vec*vec).sum().item()
                    u_norm_sq += (u*u).sum().item()
                    lam_norm_sq += (self.lambdas[count]*self.lambdas[count]).sum().item()

                pr = np.sqrt(pr)
                dr = np.sqrt(dr)
                e_pr = np.sqrt(self.dim)*abs_tol+rel_tol*max(np.sqrt(u_norm_sq), np.sqrt(self.Workers*v_norm_sq))
                e_dr = np.sqrt(self.dim)*abs_tol+rel_tol*np.sqrt(lam_norm_sq)
                
                print('primal residual: ', pr)
                print('dual residual:', dr)
                

            primal_residual.append(pr)
            dual_residual.append(dr)
            time_elapsed = time.time()-t
            print('time used: ', time_elapsed)
            print()

            if pr <= e_pr and dr <= e_dr:
                break
                
        return [self.V, self.loss, i, primal_residual, dual_residual, self.diagonalweights, self.U]

#For ACADMM, RBADMM, CADMM

def tau_update(method, i, mu_RBADMM, epsilon, C_cg, T_f, tau_list, z_old, z_pprev, z_prev, y_pprev_list, y_prev_list, y_hat_list, y_old_list,
               x_prev_list, x_old_list, primal_residual, dual_residual):

    if method == 'ACADMM':
        if i % T_f == 1:
            with torch.no_grad():
                delta_z_old = z_old - z_prev
                for count, y_hat in enumerate(y_hat_list):
                    y_hat_new = y_pprev_list[count] + tau_list[count]*(z_pprev - x_prev_list[count])
                    delta_y_hat = y_hat_new - y_hat
                    y_hat_list[count] = y_hat_new

                    delta_x_old = x_prev_list[count] - x_old_list[count]
                    x_old_list[count] = x_prev_list[count].clone().detach()

                    delta_y_old = y_prev_list[count]-y_old_list[count]
                    y_old_list[count] = y_prev_list[count].clone().detach()

                    alpha_sd = (delta_y_hat*delta_y_hat).sum().item()/(delta_x_old*delta_y_hat).sum().item()
                    alpha_mg = (delta_x_old*delta_y_hat).sum().item()/(delta_x_old*delta_x_old).sum().item()
                    beta_sd = (delta_y_old*delta_y_old).sum().item()/(delta_z_old*delta_y_old).sum().item()
                    beta_mg = (delta_z_old*delta_y_old).sum().item()/(delta_z_old*delta_z_old).sum().item()

                    alpha_hat = 1.0; beta_hat = 1.0;
                    if 2*alpha_mg > alpha_sd:
                        alpha_hat = alpha_mg
                    else:
                        alpha_hat = alpha_sd - 0.5*alpha_mg

                    if 2*beta_mg > beta_sd:
                        beta_hat = beta_mg
                    else:
                        beta_hat = beta_sd - 0.5*beta_mg

                    alpha_cor = (delta_x_old*delta_y_hat).sum().item()/(torch.norm(delta_x_old)*torch.norm(delta_y_hat)).item()
                    beta_cor = (delta_z_old*delta_y_old).sum().item()/(torch.norm(delta_z_old)*torch.norm(delta_y_old)).item()

                    tau_hat = 1.0
                    if alpha_cor > epsilon and beta_cor > epsilon:
                        tau_hat = np.sqrt(alpha_hat*beta_hat)
                    elif alpha_cor > epsilon and beta_cor <= epsilon:
                        tau_hat = alpha_hat
                    elif alpha_cor <= epsilon and beta_cor > epsilon:
                        tau_hat = beta_hat
                    else:
                        tau_hat = tau_list[count]

                    tau_new = max(min(tau_hat,(1+C_cg/(i**2))*tau_list[count]),tau_list[count]/(1+C_cg/(i**2)))
                    tau_list[count] = tau_new

    elif method=='RBADMM':
        if mu_RBADMM * np.sqrt(dual_residual) < np.sqrt(primal_residual):
            for count in range(len(tau_list)):
                tau_list[count] = 2 * tau_list[count]
                
        elif mu_RBADMM * np.sqrt(primal_residual) < np.sqrt(dual_residual):
            for count in range(len(tau_list)):
                tau_list[count] = 0.5 * tau_list[count]

def XLOSS(trainset, x, y, z, tau, M, LOSS_NAME):
    N = 1.0
    objfunc = FullLoss(trainset, x, N, M, LOSS_NAME)
    extra_terms = 0
    diff = torch.add(z, x, alpha = -1)
    normsq = pow(torch.norm(diff + y/tau), 2)
    extra_terms += 0.5*tau*normsq
    return objfunc + extra_terms

class Solve_X:
    def solve(self, x, y, z, tau, trainset, X_EPOCHS, XLOSS, M, LOSS_NAME):
        optimizer = torch.optim.LBFGS([x], max_iter=4, history_size=20 ,lr = 1e-3)
        for epoch in range(X_EPOCHS):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                loss = XLOSS(trainset, x, y, z, tau, M, LOSS_NAME) #Changed
                if loss.requires_grad:
                    loss.backward()
                return loss
            optimizer.step(closure)
        return x
        
class Combine_Y:
    def combine(self, x, y, z, tau):
        y = torch.add(y, tau*torch.add(z, x, alpha=-1))
        return y


class ADMM:
    def __init__(self, params, regularizer, rho1, rho2, trainsets, LOSS_NAME):
        self.Workers = len(params)
        self.LOSS_NAME = LOSS_NAME
        self.params = params
        dim1 = params[0].shape[0]; dim2 = params[0].shape[1]
        self.dim = 1.0*dim1*dim2
        self.regularizer = regularizer
        self.rho1 = rho1
        self.rho2 = rho2
        self.trainsets = trainsets
        self.Y = []
        self.Z = torch.zeros_like(params[0])
        self.M = [*self.Z.shape][0]
        self.Tau = []
        self.X_EPOCHS = 0

    def solve_x(self, i, M, LOSS_NAME):
        solve = Solve_X()
        return solve.solve(self.params[i], self.Y[i], self.Z, self.Tau[i], self.trainsets[i], self.X_EPOCHS, XLOSS, M, LOSS_NAME)

    def solve_z(self, rho1, rho2, U, lambdas, tau_list):
        threshold = torch.nn.Threshold(0,0)
        X = torch.zeros_like(U[0])
        K = rho2
        for count, u in enumerate(U):
            X = X + tau_list[count]*u.clone()-lambdas[count]
            K = K + tau_list[count]
        value = rho1
        V = 1.0/K*threshold(abs(X)-value)*torch.sign(X)
        return V

    def combine_y(self, i):
        combine = Combine_Y()
        return combine.combine(self.params[i], self.Y[i], self.Z, self.Tau[i])

    def do(self, maxiter=1, X_EPOCHS=3, abs_tol=1e-5, rel_tol=1e-4, epsilon=0.2, T_f=2, C_cg=1e10, \
           closure=None, start_tau=1, method='ACADMM'):
        self.X_EPOCHS = X_EPOCHS
        M = self.M
        LOSS_NAME = self.LOSS_NAME

        #Data_Tracking
        iteration_number = []
        z_loss = []
        f_values = []
        g_values = []
        objfunc = []
        p_res = []
        d_res = []
        y_hat_list = []
        x_old_list = []
        y_old_list = []
        Y_previous = []
        
        z_old = self.Z.clone().detach()
        for x in self.params:
            self.Y.append(torch.zeros_like(x))
            self.Tau.append(start_tau)
            y_hat_list.append(torch.zeros_like(x))
            x_old_list.append(x.clone().detach())
            y_old_list.append(torch.zeros_like(x))
            Y_previous.append(torch.zeros_like(x))

        for i in range(maxiter):
            t_opt = time.time()

            #X_updates
            x_previous = []
            for count in range(self.Workers):
                x_previous.append(self.params[count].clone().detach())
                self.params[count] = self.solve_x(count, M, LOSS_NAME)

            with torch.no_grad():
                #Z_update
                if i > 0:
                    z_pprev = z_previous.clone().detach()
                z_previous = (self.Z).clone().detach()
                self.Z = self.solve_z(self.rho1, self.rho2, self.params, self.Y, self.Tau)

                #Y_update
                Y_pprev = []
                for count in range(self.Workers):
                    if i > 0:
                        Y_pprev.append(Y_previous[count].clone().detach())
                    Y_previous[count] = self.Y[count].clone().detach()
                    self.Y[count] = self.combine_y(count)

                #Stopping Criteria
                pr = 0
                dr = 0
                dr_rb = 0
                x_norm_sq = 0
                z_norm_sq = (self.Z*self.Z).sum().item()
                y_norm_sq = 0
                
                for count, x in enumerate(self.params):
                    pr += ((self.Z-x)*(self.Z-x)).sum().item()
                    vec = z_previous - self.Z
                    dr += (vec*vec).sum().item()
                    vec_rb = self.Tau[count]*(z_previous-self.Z)
                    dr_rb += (vec_rb*vec_rb).sum().item()
                    x_norm_sq += (x*x).sum().item()
                    y_norm_sq += (self.Y[count]*self.Y[count]).sum().item()

                pr = np.sqrt(pr)
                dr = np.sqrt(dr)
                e_pr = np.sqrt(self.dim)*abs_tol+rel_tol*max(np.sqrt(x_norm_sq), np.sqrt(self.Workers*z_norm_sq))
                e_dr = np.sqrt(self.dim)*abs_tol+rel_tol*np.sqrt(y_norm_sq)
                
                #Data Tracking
                p_res.append(np.sqrt(pr))
                d_res.append(np.sqrt(dr))
                z_diff = torch.norm(z_previous - self.Z)
                z_accuracy = torch.norm(z_previous - self.Z)/torch.norm(z_previous)
                iteration_number.append(i+1)
                z_loss.append(z_accuracy.item())

                #f(z) + g(z)
                f_and_g = 0
                for count, trainset in enumerate(self.trainsets):
                    f_and_g += FullLoss(trainset, self.Z, self.Workers, M, LOSS_NAME)
                f_v = f_and_g.item()
                f_values.append(f_v)
                g_v = self.regularizer(self.Z).item()
                g_values.append(g_v)
                f_and_g = f_v + g_v
                objfunc.append(f_and_g)

                optimize_time = time.time()-t_opt
                
                if pr <= e_pr and dr <= e_dr:
                    break

                #tau Updates
                t_tau = time.time()

                if method == 'RBADMM' and i==0:
                    if 10 * np.sqrt(dr_rb) < np.sqrt(pr):
                        for count in range(len(self.Tau)):
                            self.Tau[count] = 2 * self.Tau[count]
                
                    elif 10 * np.sqrt(pr) < np.sqrt(dr_rb):
                        for count in range(len(self.Tau)):
                            self.Tau[count] = 0.5 * self.Tau[count]

                if i > 0:
                    tau_update(method, i+1, 10, epsilon, C_cg, T_f, self.Tau, z_old, z_pprev, z_previous, Y_pprev, Y_previous, y_hat_list, y_old_list,
                              x_previous, x_old_list, pr, dr_rb)

                if (i+1) % T_f == 1:
                    z_old = (z_previous).clone().detach()

                tau_update_time = time.time()-t_tau


            total_time = optimize_time+tau_update_time

            print('%d      %.2e       %.2e       %.2e       %.2e'
                        % (i+1, f_and_g, pr, dr, total_time))
            print()

        return [self.Z,i+1,iteration_number, z_loss, f_values, g_values, objfunc, p_res, d_res, total_time, self.params]

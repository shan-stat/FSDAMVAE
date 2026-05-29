import os
import torch
import numpy as np
import pandas
from torch import nn
import torch.nn.functional as F
import datetime
import timeit 
#from sklearn.preprocessing import scale

opt_str="adam" # adam simple
reltol = 1e-8 # if earlytop, this is used, has a huge impact on the results
stop_if_loss_incr_thrsh=50

loss_fn = nn.MSELoss(reduction='mean') 
learning_rate = 1e-4 # 1e-4, 1e-5 is too slow


def main(dat, opt_numCode, opt_seed, opt_model, opt_gpu, opt_k, opt_nEpochs, opt_constr, opt_tuneParam, opt_klParam, opt_penfun, opt_ortho, opt_earlystop, verbose):        
    # new argument 
    # opt_klParam: tuning parameter for KL loss
    
    if torch.cuda.is_available() and opt_gpu >= 0:
        device = torch.device("cuda:"+str(opt_gpu))
    else :
        device = torch.device("cpu")        
    dat = torch.tensor(dat.values, dtype=torch.float, device=device) # dtype is needed here otherwise will get an error in training: Expected object of scalar type Float but got scalar type Double for argument #4 'mat1'
    
    N=dat.shape[0]; n=N
    p=dat.shape[1]        
    
    torch.manual_seed(opt_seed)
    out=[]
    x_new = dat
    allcodes=torch.zeros([n,0], dtype=torch.float, device=device)
    
    if opt_numCode<0: opt_numCode=p
    for i in range(opt_numCode) :
        if opt_model == 'o':
            # ==============================================================================
            # MODIFIED: Shifted the residual subtraction index from out[i-1][2] to out[i-1][3]
            # because AE1 now returns an extra element (kl_loss_value) ahead of y_pred.
            # ==============================================================================
            # if i>0 : x_new = x_new - out[i-1][2]
            if i>0 : x_new = x_new - out[i-1][3]
            
        # opt_klParam added; 
        out.append (AE1(x_new, i, allcodes, device, opt_model, opt_k, opt_nEpochs, opt_constr, opt_tuneParam, opt_klParam, opt_penfun, opt_ortho, opt_earlystop, verbose))
        # allcodes=torch.cat([allcodes, out[i][3]],1)
        allcodes=torch.cat([allcodes, out[i][4]],1)
        
    return out


# find one nonlinear component
def AE1(x, stage, prev_codes, device, opt_model, opt_k, opt_nEpochs, opt_constr, opt_tuneParam, opt_klParam, opt_penfun, opt_ortho, opt_earlystop, verbose):    
    # new argument
    # opt_klParam: tuning parameter for KL loss
    
    p=x.shape[1]
    
    if opt_model == 'o':
        model = Model_Old(opt_k, p)
    elif opt_model == 'n':
        model = Model_New(opt_k, p, stage)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    y=x # y is output
    
    loss_old = np.inf
    cnt=0
    history=[]
    if verbose : print("iter", " reconstruct_loss", " kl_loss", " penalized loss") # kl_loss added
    
    for t in range(opt_nEpochs):
        if opt_model == 'o':
            y_pred, code = model(x)
        elif opt_model == 'n':
            y_pred, code = model(x, prev_codes)
            
        loss = loss_fn(y_pred, y)
        reconstruct_loss = loss.item()

        # ========= NEW: Calculate and Add KL Divergence Regularization Penalty =========
        # formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_elementwise = -0.5 * (1 + model.logvar - model.mu.pow(2) - model.logvar.exp())
        
        if opt_penfun == 'sum':
            current_kl = kl_elementwise.sum()
            loss += opt_klParam * current_kl
        elif opt_penfun == 'mean':
            current_kl = kl_elementwise.mean()
            loss += opt_klParam * current_kl
            
        kl_loss = current_kl.item()
        # ===============================================================================
        
        if opt_constr=="penalization":
            # penalize negtive gradients
            for j in range(p) :
                d = torch.autograd.grad(y_pred[:, j].sum(), code, retain_graph=True, create_graph=True)[0]
                loss += opt_tuneParam * F.relu(-d).sum() 
        
        elif opt_constr=="newpenalization":
            # in the first stage, manually compute derivatives at a grid of values in the range of the code
            # in the later stages, same as penalization
            d1 = model.output_code_deriv(stage==0, device)
            if opt_penfun == 'sum':
                loss += opt_tuneParam * F.relu(-d1).sum()
            elif opt_penfun == 'mean':
                loss += opt_tuneParam * F.relu(-d1).mean()
        
        if opt_ortho>0:
            # penalize covariance
            prev_means = [prev_codes[:,j].mean() for j in range(stage)]
            mean_code = code[:,0].mean()
            for j in range(stage) :
                loss += opt_ortho * abs((code[:,0] * prev_codes[:,j]).mean() - mean_code * prev_means[j])        
      
        if t % 100 == 0:
            if verbose : print(t, reconstruct_loss, kl_loss, loss.item()) # kl_loss added
            history.append(loss.item())
        
        if opt_earlystop=="yes":
            if (abs(loss_old-loss.item())/loss_old<reltol):
                if verbose : print("reltol reached")
                break    
    
        if (loss.item()>loss_old) :
            cnt+=1
            if(cnt == stop_if_loss_incr_thrsh) :
                if verbose : print("loss starts increased for "+str(stop_if_loss_incr_thrsh) + " times")
                break
        else :
            cnt=0
        loss_old=loss.item()
            
        if (opt_str=="simple") :
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
        
        elif (opt_str=="adam") :
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (opt_constr=="constrained"):
                with torch.no_grad():
                    iter=0
                    for value in model.parameters():
                        iter += 1
                        if (iter==5 or iter==7):
                            value.data.clamp_(0)
    
    
#    from matplotlib import pyplot as plt
#    plt.figure(figsize=(30,8))
#    for i in range(4):
#        plt.subplot(2,4,i+1)
#        plt.scatter(code.detach().cpu().numpy(), y_pred[:,i-1].detach().cpu().numpy(), s=1)
#        plt.subplot(2,4,i+5)
#        plt.plot(range(d1.shape[1]), d1.detach().cpu().numpy()[i,])
#        plt.axhline(y=0, lw=0.5)
#    plt.savefig("btmp.pdf"); plt.close()
    
    decoder_w=torch.cat((model.demap[0].weight, model.output[0].weight.t()),1)
    decoder_b=torch.cat((model.demap[0].bias,   model.output[0].bias),0)
    # the following line matches y_pred
    #torch.mm((torch.mm(code, model.top[0].weight.t()) + model.top[0].bias).tanh(), model.top[2].weight.t()) + model.top[2].bias
    
    # kl_loss added
    return reconstruct_loss, kl_loss, loss.item(), y_pred.detach(), code.detach(), decoder_w.detach().cpu(), decoder_b.detach().cpu(), history


class Model_New (nn.Module):
    def __init__(self, k, p, stage):
        super().__init__()
        # FS-DAM
        # self.bottom = nn.Sequential(nn.Linear(p, k), nn.Tanh(), nn.Linear(k, 1))
        
        # FS-DAM + VAE
        # ========= MODIFIED: Separation of Encoder backbone and Latent Parameters =========
        self.encoder_backbone = nn.Sequential(nn.Linear(p, k), nn.Tanh())
        self.fc_mu = nn.Linear(k, 1)      # Squeezes to 1D mean
        self.fc_logvar = nn.Linear(k, 1)  # Squeezes to 1D log-variance
        # ===================================================================================
        
        self.demap  = nn.Sequential(nn.Linear(stage+1, k), nn.Tanh())
        self.output = nn.Sequential(nn.Linear(k, p))

    # ========= NEW: Reparameterization Trick Function =========
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar/2) # exp(logvar/2) = sigma
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Use the deterministic mean value during inference/evaluation
    # ==========================================================
        
    def forward(self, x, prev_codes):
        self.n = x.shape[0]
        # FS-DAM
        # self.code = self.bottom(x)        
        
        # FS-DAM + VAE
        # ========= MODIFIED: VAE Forward Path Flow =========
        hidden = self.encoder_backbone(x)
        self.mu = self.fc_mu(hidden)
        self.logvar = self.fc_logvar(hidden)
        self.code = self.reparameterize(self.mu, self.logvar)
        # ====================================================        
        
        self.z = self.demap(torch.cat([self.code, prev_codes],1)) # not saving self.z does not speed it up
        return self.output(self.z), self.code
        
    def output_code_deriv(self, to_sample, device):
        if to_sample :
            gridsize = self.n * 2
            code_grid = torch.linspace(min(self.code).item(), max(self.code).item(), steps=gridsize, dtype=torch.float, 
                                       device=device, requires_grad=False).view(gridsize, 1)
            z=self.demap(code_grid)
        else :
            z=self.z # computing this derivative ourselves is faster than using the autograd
        
        tmp = (1-z.pow(2)) * self.demap[0].weight[:,0]
        return (torch.mm(tmp, self.output[0].weight.t()))


class Model_Old (nn.Module):
    def __init__(self, k, p):
        super().__init__()
        # FS-DAM
        #self.bottom = nn.Sequential(nn.Linear(p, k), nn.Tanh(), nn.Linear(k, 1))
        ##self.top = nn.Sequential(nn.Linear(1, k), nn.Tanh(), nn.Linear(k, p))
        ## top is split into demap and output

        # FS-DAM + VAE
        # ========= MODIFIED: Separation of Encoder backbone and Latent Parameters =========
        self.encoder_backbone = nn.Sequential(nn.Linear(p, k), nn.Tanh())
        self.fc_mu = nn.Linear(k, 1)      # Squeezes to 1D mean
        self.fc_logvar = nn.Linear(k, 1)  # Squeezes to 1D log-variance
        # ===================================================================================
        
        self.demap  = nn.Sequential(nn.Linear(1, k), nn.Tanh())
        self.output = nn.Sequential(nn.Linear(k, p))

    # ========= NEW: Reparameterization Trick Function =========
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar/2) # exp(logvar/2) = sigma
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Use the deterministic mean value during inference/evaluation
    # ==========================================================
    
    def forward(self, x):
        self.n = x.shape[0]
        # FS-DAM
        # self.code = self.bottom(x)

        # FS-DAM + VAE
        # ========= MODIFIED: VAE Forward Path Flow =========
        hidden = self.encoder_backbone(x)
        self.mu = self.fc_mu(hidden)
        self.logvar = self.fc_logvar(hidden)
        self.code = self.reparameterize(self.mu, self.logvar)
        # ====================================================        

        self.z = self.demap(self.code) # not saving self.z does not speed it up
        return self.output(self.z), self.code
    
    def output_code_deriv(self, to_sample, device):
        if to_sample :
            gridsize = self.n * 2
            code_grid = torch.linspace(min(self.code).item(), max(self.code).item(), steps=gridsize, dtype=torch.float, 
                                       device=device, requires_grad=False).view(gridsize, 1)
            z=self.demap(code_grid)
        else :
            z=self.z # computing this derivative ourselves is faster than using the autograd
        
#        print(z.shape)
#        print(self.demap[0].weight.shape)
        tmp = (1-z.pow(2)) * self.demap[0].weight[:,0]
        return (torch.mm(tmp, self.output[0].weight.t()))

import numpy as np
import sys, os, time
import torch 
import torch.nn as nn
import data
import architecture
import optuna
import utils as U

# get optuna parameters
def read_database(storage, study_name, model_rank):

    # load the optuna study
    study = optuna.load_study(study_name=study_name, storage=storage)

    # get the scores of the study trials
    values = np.zeros(len(study.trials))
    completed = 0
    for i,t in enumerate(study.trials):
        values[i] = t.value
        if t.value is not None:  completed += 1

    # get the info of the best trial
    indexes = np.argsort(values)
    for i in [model_rank]: 
        trial = study.trials[indexes[i]]
        print("\nTrial number {}".format(trial.number))
        print("Value: %.5e"%trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        num    = trial.number 
        dr     = trial.params['dr']
        hidden = trial.params['hidden']
        lr     = trial.params['lr']
        wd     = trial.params['wd']

    return num, dr, hidden, lr, wd

# load model
def load_model(storage, study_name, model_rank, root_models, suffix, 
               arch, channels, device):

    # get the best trial from the database
    num, dr, hidden, lr, wd = read_database(storage, study_name, model_rank)

    # get the name of the file with the network weights
    #fmodel = '%s/weights_Nbody_Mtot_%d_%s.pt'%(root_models, num, suffix)
    fmodel = '%s/model_%d_%s.pt'%(root_models, num, suffix)

    # load the model weights
    model = architecture.get_architecture(arch+'_err', hidden, dr, channels)
    #model = nn.DataParallel(model)
    model.to(device=device)
    network_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters in the model = %d'%network_total_params)

    # load best-model, if it exists
    if os.path.exists(fmodel):  
        print('Loading model...')
        model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
    else:
        print(fmodel)
        raise Exception('model doesnt exists!!!')
    
    return model


# test the model
def test_model(test_loader, model, device, fresults):

    # get the number of maps in the test set
    num_maps = U.dataloader_elements(test_loader)
    print('\nNumber of maps in the test set: %d'%num_maps)

    # define the arrays containing the value of the parameters
    params_true = np.zeros((num_maps,1), dtype=np.float32)
    params_NN   = np.zeros((num_maps,1), dtype=np.float32)
    errors_NN   = np.zeros((num_maps,1), dtype=np.float32)

    # get test loss
    g = [0]
    test_loss1 = torch.zeros(len(g)).to(device)
    test_loss2 = torch.zeros(len(g)).to(device)
    test_loss, points = 0.0, 0
    model.eval()
    for x, y in test_loader:
        print(points)
        with torch.no_grad():
            bs    = x.shape[0]    #batch size
            x     = x.to(device)  #send data to device
            y     = y.to(device)  #send data to device
            p     = model(x)      #prediction for mean and variance
            y_NN  = p[:,:6]       #prediction for mean
            e_NN  = p[:,6:]       #prediction for error
            loss1 = torch.mean((y_NN[:,g] - y[:,g])**2,                     axis=0)
            loss2 = torch.mean(((y_NN[:,g] - y[:,g])**2 - e_NN[:,g]**2)**2, axis=0)
            test_loss1 += loss1*bs
            test_loss2 += loss2*bs

            # save results to their corresponding arrays
            params_true[points:points+x.shape[0]] = y[:,g].cpu().numpy() 
            params_NN[points:points+x.shape[0]]   = y_NN[:,g].cpu().numpy()
            errors_NN[points:points+x.shape[0]]   = e_NN[:,g].cpu().numpy()
            points    += x.shape[0]
    test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss = %.3e\n'%test_loss)

    Norm_error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    print('Normalized Error A = %.3f'%Norm_error[0])

    # de-normalize
    minimum = np.array([0.6])
    maximum = np.array([1.4])
    params_true = params_true*(maximum - minimum) + minimum
    params_NN   = params_NN*(maximum - minimum) + minimum
    errors_NN   = errors_NN*(maximum - minimum)

    error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    print('Error A = %.3f'%error[0])

    mean_error = np.absolute(np.mean(errors_NN, axis=0))
    print('Bayesian error A = %.3f'%mean_error[0])

    rel_error = np.sqrt(np.mean((params_true - params_NN)**2/params_true**2, axis=0))
    print('Relative error A = %.3f'%rel_error[0])

    # save results to file
    dataset = np.zeros((num_maps,3), dtype=np.float32)
    dataset[:,0] = params_true[:,0]
    dataset[:,1] = params_NN[:,0]
    dataset[:,2] = errors_NN[:,0]
    np.savetxt(fresults,  dataset)



######################################### INPUT #######################################
root_models = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/Results/models_Gaussian'
arch        = 'o3'
suffix      = '%s_smoothing_0'%arch
channels    = 1
storage     = 'sqlite:////mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/databases/Gaussian_LH_o3_smoothing_0.db'
study_name = 'wd_dr_hidden_lr_%s'%arch
model_rank = 0 #0 is the best model, 1 is the second best model,...etc

# data parameters
mode            = 'test'
seed            = 1
batch_size      = 32
splits          = 1
monopole_train  = True
monopole_test   = True
just_monopole   = False
smoothing_test  = 0
smoothing_train = 0

# data parameters II
root_maps   = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps'
sim_train   = 'Gaussian'
sim_test    = 'Gaussian'
z           = 0.0
f_maps      = ['%s/maps_%s/Images_M_%s_LH_z=%.2f.npy'%(root_maps,sim_test,sim_test,z)]
f_params    = '%s/maps_%s/params_%s.txt'%(root_maps,sim_test,sim_test)
f_maps_norm = ['%s/maps_%s/Images_M_%s_LH_z=%.2f.npy'%(root_maps,sim_train,sim_train,z)]

# results
fresults = 'Trained_%s_tested_%s_z=%.2f.txt'%(sim_train,sim_test,z)
#######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# load the model
model = load_model(storage, study_name, model_rank, root_models, suffix, 
                   arch, channels, device)

# get the test data
test_loader = data.create_dataset_multifield(mode, seed, f_maps, f_params, 
                            batch_size, splits, f_maps_norm, monopole=monopole_test, 
                            monopole_norm=monopole_train, rot_flip_in_mem=True, 
                            shuffle=False, just_monopole=just_monopole, 
                            smoothing=smoothing_test, smoothing_norm=smoothing_train, 
                            verbose=True, Gaussian=True)

# test the model on the data
test_model(test_loader, model, device, fresults)

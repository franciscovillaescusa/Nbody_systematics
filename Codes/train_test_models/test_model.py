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
    fmodel = '%s/weights_Nbody_Mtot_%d_%s.pt'%(root_models, num, suffix)

    # load the model weights
    model = architecture.get_architecture(arch+'_err', hidden, dr, channels)
    model = nn.DataParallel(model)
    model.to(device=device)
    network_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters in the model = %d'%network_total_params)

    # load best-model, if it exists
    if os.path.exists(fmodel):  
        print('Loading model...')
        model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
    else:
        raise Exception('model doesnt exists!!!')
    
    return model


# test the model
def test_model(test_loader, model, device, fresults):

    # get the number of maps in the test set
    num_maps = U.dataloader_elements(test_loader)
    print('\nNumber of maps in the test set: %d'%num_maps)

    # define the arrays containing the value of the parameters
    params_true = np.zeros((num_maps,6), dtype=np.float32)
    params_NN   = np.zeros((num_maps,6), dtype=np.float32)
    errors_NN   = np.zeros((num_maps,6), dtype=np.float32)

    # get test loss
    g = [0, 1, 2, 3, 4, 5]
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
            params_true[points:points+x.shape[0]] = y.cpu().numpy() 
            params_NN[points:points+x.shape[0]]   = y_NN.cpu().numpy()
            errors_NN[points:points+x.shape[0]]   = e_NN.cpu().numpy()
            points    += x.shape[0]
    test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss = %.3e\n'%test_loss)

    Norm_error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    print('Normalized Error Omega_m = %.3f'%Norm_error[0])
    print('Normalized Error sigma_8 = %.3f'%Norm_error[1])
    print('Normalized Error A_SN1   = %.3f'%Norm_error[2])
    print('Normalized Error A_AGN1  = %.3f'%Norm_error[3])
    print('Normalized Error A_SN2   = %.3f'%Norm_error[4])
    print('Normalized Error A_AGN2  = %.3f\n'%Norm_error[5])

    # de-normalize
    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    params_true = params_true*(maximum - minimum) + minimum
    params_NN   = params_NN*(maximum - minimum) + minimum
    errors_NN   = errors_NN*(maximum - minimum)

    error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    print('Error Omega_m = %.3f'%error[0])
    print('Error sigma_8 = %.3f'%error[1])
    print('Error A_SN1   = %.3f'%error[2])
    print('Error A_AGN1  = %.3f'%error[3])
    print('Error A_SN2   = %.3f'%error[4])
    print('Error A_AGN2  = %.3f\n'%error[5])

    mean_error = np.absolute(np.mean(errors_NN, axis=0))
    print('Bayesian error Omega_m = %.3f'%mean_error[0])
    print('Bayesian error sigma_8 = %.3f'%mean_error[1])
    print('Bayesian error A_SN1   = %.3f'%mean_error[2])
    print('Bayesian error A_AGN1  = %.3f'%mean_error[3])
    print('Bayesian error A_SN2   = %.3f'%mean_error[4])
    print('Bayesian error A_AGN2  = %.3f\n'%mean_error[5])

    rel_error = np.sqrt(np.mean((params_true - params_NN)**2/params_true**2, axis=0))
    print('Relative error Omega_m = %.3f'%rel_error[0])
    print('Relative error sigma_8 = %.3f'%rel_error[1])
    print('Relative error A_SN1   = %.3f'%rel_error[2])
    print('Relative error A_AGN1  = %.3f'%rel_error[3])
    print('Relative error A_SN2   = %.3f'%rel_error[4])
    print('Relative error A_AGN2  = %.3f\n'%rel_error[5])

    # save results to file
    dataset = np.zeros((num_maps,18), dtype=np.float32)
    dataset[:,:6]   = params_true
    dataset[:,6:12] = params_NN
    dataset[:,12:]  = errors_NN
    np.savetxt(fresults,  dataset)



######################################### INPUT #######################################
# model parameters
root        = '/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/2D_maps'
root_models = '%s/inference/weights'%root
arch        = 'o3'
suffix      = 'all_steps_500_500_%s'%arch
channels    = 1

# optuna parameters
storage    = 'sqlite:///%s/inference/databases/Nbody_%s_Mtot_%s.db'%(root,arch,suffix)
study_name = 'wd_dr_hidden_lr_%s'%arch
model_rank = 0 #0 is the best model, 1 is the second best model,...etc

# data parameters
mode            = 'all'
seed            = 1
batch_size      = 64
splits          = 15
monopole_train  = True
monopole_test   = True
just_monopole   = False
smoothing_test  = 0
smoothing_train = 0

# data parameters II
sim         = 'Ramses'
redshift    = 0.0
root_data   = '/mnt/ceph/users/fvillaescusa/Nbody_systematics/maps/maps_%s'%sim
f_maps      = ['%s/Images_M_%s_LH_z=%.2f.npy'%(root_data,sim,redshift)]
f_params    = '%s/params_%s.txt'%(root_data,sim)
#f_maps      = ['%s/data/Maps_Mtot_Nbody_IllustrisTNG_LH_z=0.0.npy'%root]
#f_params    = '%s/data/params_Nbody_IllustrisTNG.txt'%root
f_maps_norm = ['%s/data/Maps_Mtot_Nbody_IllustrisTNG_LH_z=0.0.npy'%root]

# results
fresults = 'Trained_Gadget_tested_%s_z=%.2f.txt'%(sim,redshift)
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
                            verbose=True)

# test the model on the data
test_model(test_loader, model, device, fresults)

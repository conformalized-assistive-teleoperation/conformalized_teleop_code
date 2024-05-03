import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_telop_controller_with_quantiles(model_architecture, remapping_train_loader, qr_criterion, state_dim,
            action_dim, latent_dim, hidden_dim, bsz, lr, epochs, experiment_name, to_save=False):


    # Create Remapping Model
    remapping_model = model_architecture(action_dim, state_dim, latent_dim, hidden_dim)

    # Create Optimizer
    optimizer = torch.optim.Adam(remapping_model.parameters(), lr=lr)

    # Create lists to store losses
    all_qlo_losses = []
    all_qhi_losses = []
    all_mse_losses = []
    epoch_losses = []
    train_losses = []

    # Start Training
    for epoch in range(epochs):
        print("epoch", epoch)
        epoch_loss = 0
        for batch in remapping_train_loader:
            (state, low_dim_action, true_high_dim_action) = batch

            # Forward Pass
            predicted_action_with_quantiles = remapping_model.forward(state, low_dim_action)

            # split into 3: lower quantile, mean prediction, upper quantile
            split_action = torch.split(predicted_action_with_quantiles, action_dim, dim=1)
            action_quant_low = split_action[0]
            action_med = split_action[1]
            action_quant_high = split_action[2]

            # measure loss
            quantile_loss, loss_components = qr_criterion(action_med, action_quant_low, action_quant_high,
                                                          true_high_dim_action, F.mse_loss)

            # record loss for this batch
            loss = quantile_loss
            (qlo_loss, qhi_loss, mse_loss) = loss_components
            # detach all the losses and append to list
            all_qlo_losses.append(qlo_loss.item())
            all_qhi_losses.append(qhi_loss.item())
            all_mse_losses.append(mse_loss.item())
            train_losses.append(loss.item())
            epoch_loss += loss.item()

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print epoch loss
        print("total epoch_loss = ", epoch_loss)
        epoch_losses.append(epoch_loss)

    if to_save:
        # Save Final Model
        torch.save(remapping_model.state_dict(), f"results/{experiment_name}/models/epoch_final.pt")
        print("Saved Model!")
        # save losses
        with open(f"results/{experiment_name}/models/epoch_losses.pkl", "wb") as f:
            pickle.dump(epoch_losses, f)
        with open(f"results/{experiment_name}/models/train_losses.pkl", "wb") as f:
            pickle.dump(train_losses, f)

        # Plot Losses
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot([elem for elem in epoch_losses])
        plt.title("Epoch Losses")
        plt.savefig(f"results/{experiment_name}/models/epoch_losses.png")
        plt.close()

        # plot qlo losses
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Q_Low Loss')
        plt.plot([elem for elem in all_qlo_losses])
        plt.title("Q_Low Losses")
        plt.savefig(f"results/{experiment_name}/models/qlo_losses.png")
        plt.close()

        # plot qhi losses
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Q_High Loss')
        plt.plot([elem for elem in all_qhi_losses])
        plt.title("Q_High Losses")
        plt.savefig(f"results/{experiment_name}/models/qhi_losses.png")
        plt.close()

        # plot mse losses
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('MSE Loss')
        plt.plot([elem for elem in all_mse_losses])
        plt.title("MSE Losses")
        plt.savefig(f"results/{experiment_name}/models/mse_losses.png")
        plt.close()


def train_ensemble_teleop_controller(model_architecture, remapping_train_loaders, ensemble_criterion, state_dim,
            action_dim, latent_dim, hidden_dim, bsz, lr, epochs, experiment_name, to_save=False):

    # Create Remapping Model - 5 Ensemble models
    remapping_model_1 = model_architecture(action_dim, state_dim, latent_dim, hidden_dim)
    optimizer_1 = torch.optim.Adam(remapping_model_1.parameters(), lr=lr)

    remapping_model_2 = model_architecture(action_dim, state_dim, latent_dim, hidden_dim)
    optimizer_2 = torch.optim.Adam(remapping_model_2.parameters(), lr=lr)

    remapping_model_3 = model_architecture(action_dim, state_dim, latent_dim, hidden_dim)
    optimizer_3 = torch.optim.Adam(remapping_model_3.parameters(), lr=lr)

    remapping_model_4 = model_architecture(action_dim, state_dim, latent_dim, hidden_dim)
    optimizer_4 = torch.optim.Adam(remapping_model_4.parameters(), lr=lr)

    remapping_model_5 = model_architecture(action_dim, state_dim, latent_dim, hidden_dim)
    optimizer_5 = torch.optim.Adam(remapping_model_5.parameters(), lr=lr)

    list_of_remapping_models = [remapping_model_1, remapping_model_2, remapping_model_3, remapping_model_4,
                                remapping_model_5]
    list_of_optimizers = [optimizer_1, optimizer_2, optimizer_3, optimizer_4, optimizer_5]



    for epoch in range(epochs):
        print("epoch", epoch)
        epoch_loss = 0
        for remapping_idx in range(len(list_of_remapping_models)):
            remapping_train_loader = remapping_train_loaders[remapping_idx]
            remapping_model = list_of_remapping_models[remapping_idx]
            optimizer = list_of_optimizers[remapping_idx]

            for batch in remapping_train_loader:
            # print("len batch", len(batch))
                (state, low_dim_action, true_high_dim_action) = batch
                predicted_action_with_quantiles = remapping_model.forward(state, low_dim_action)

                # split into 3
                split_action = torch.split(predicted_action_with_quantiles, action_dim, dim=1)
                # take the middle part
                action_med = split_action[0]
                action_var = split_action[1]

                quantile_loss, loss_components = ensemble_criterion(action_med,
                                                                     action_var,
                                                                     true_high_dim_action,
                                                                     F.mse_loss)
                loss = quantile_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Save Final Model
    if to_save:
        for remapping_idx in range(len(list_of_remapping_models)):
            remapping_model = list_of_remapping_models[remapping_idx]
            torch.save(remapping_model.state_dict(),
                       f"results/{experiment_name}/models/remapping_{remapping_idx}_epoch_final.pt")
            print("Saved Model!")



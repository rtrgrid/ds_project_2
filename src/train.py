import torch

def train_model(model, X_train, y_train, device, epochs=50, lr=1e-3):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.to(device)

    batch_size = X_train.shape[0]

    # Create dummy time features (all zeros)
    past_time_features = torch.zeros(
        (batch_size, X_train.shape[1], 1),
        dtype=torch.float32
    ).to(device)

    # Mask (1 means observed)
    past_observed_mask = torch.ones_like(X_train).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(
            past_values=X_train,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
        )

        preds = outputs.prediction_outputs
        loss = loss_fn(preds, y_train)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model

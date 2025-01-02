import tensorflow as tf
from collections import defaultdict
from modules.training_helper import evaluate_loss

def train_model(
    model,
    datasets,
    summary_writer,
    saving_path,
    evaluate_freq,
    max_epoch,
    loss_name,
    L_rate,
    overfit_stop=None
):
    if loss_name == 'MSE':
        loss_function = tf.keras.losses.MeanSquaredError()
    elif loss_name == 'MAE':
        loss_function = tf.keras.losses.MeanAbsoluteError()

        
    optimizer = tf.keras.optimizers.Adam( float(L_rate) )
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    @tf.function
    def train_step(data_in, training=True):
        with tf.GradientTape() as tape:
            data_out = model(data_in, training=training)
            pred_loss = loss_function(data_in, data_out)
        gradients = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        avg_losses[f'{loss_name}'].update_state(pred_loss)
        return
    

    best_loss = 9e10
    best_epoch = 0
    print('Epoch start.')
    for epoch in range(1, max_epoch+1):
        # ---- train
        for data_in in datasets['train']:
            train_step(data_in, training=True)

        for loss_name, avg_loss in avg_losses.items():
            with summary_writer.as_default():
                tf.summary.scalar(loss_name, avg_loss.result(), step=epoch)
            avg_loss.reset_states()

        # ---- evaluate
        if (epoch) % evaluate_freq == 0:
            print(f'Completed epoch {epoch}. Do evaluation.')
            
            for phase in ['valid']:
                loss  = evaluate_loss(model, datasets[phase], loss_function)
                with summary_writer.as_default():
                    tf.summary.scalar(f'[{phase}]: {loss_name}', loss, step=epoch)
            
            valid_loss = loss
            if best_loss >= valid_loss:
                best_loss = valid_loss
                best_epoch = epoch
                print(f'Get the best loss so far at epoch {epoch}! Saving the model.')
                model.save_weights(f'{saving_path}/AE', save_format='tf')
            elif overfit_stop and (epoch - best_epoch) >= overfit_stop:
                print('Reach the overfitting stop.')
                break
                


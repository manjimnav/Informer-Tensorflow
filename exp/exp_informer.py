from data.data_loader import DatasetETT
from models.model import Informer
import tensorflow as tf
import time
import os


class ExpInformer():
    def __init__(self, args):
        super(ExpInformer, self)
        self.args = args
        self.model = self._build_model()

    def _build_model(self):

        model = Informer(
            self.args.enc_in,
            self.args.dec_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.batch_size,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.embed,
            self.args.data[:-1],
            self.args.activation
        )

        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        dataset = DatasetETT(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            shuffle=shuffle_flag,
            drop_last=drop_last,
            batch_size=batch_size,
            is_minute=self.args.data == 'ETTm1'
        )

        print(flag, len(dataset))

        return dataset

    def _select_optimizer(self):
        model_optim = tf.keras.optimizers.Adam(lr=self.args.learning_rate)
        return model_optim

    def train(self, setting):
        train_data = self._get_data(flag='train')
        valid_data = self._get_data(flag='val')

        train_steps = len(train_data)
        valid_steps = len(valid_data)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self.args.patience, restore_best_weights=True)

        checkpoint_path = f'./checkpoints/{setting}'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        model_optim = self._select_optimizer()
        loss = tf.keras.losses.mse

        self.model.compile(optimizer=model_optim, loss=loss, metrics=['mse', 'mae'],
                           run_eagerly=True)

        self.model.fit(train_data.get_dataset(),
                       steps_per_epoch=train_steps,
                       validation_data=valid_data.get_dataset(),
                       validation_steps=valid_steps,
                       callbacks=[early_stopping],
                       epochs=self.args.train_epochs)

        self.model.save_weights(checkpoint_path+'/model.h5')

        return self.model

    def test(self, setting):
        test_data = self._get_data(flag='test')

        loss, mse, mae = self.model.evaluate_generator(test_data.get_dataset(), len(test_data))

        print('mse:{}, mae:{}'.format(mse, mae))

import torch
import torch.nn as nn

class FeatureRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=3,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
        use_lstm=False,
        lstm_input_dim=None,
        lstm_hidden=64,
        lstm_layers=1,
        bidirectional=False
    ):
        """
        Parameters
        ----------
        input_dim      : int
            Number of features per sample
        output_dim     : int
            Number of  targets (e.g., 3 for H, Ra, Xa)
        hidden_dim     : int
            Hidden size for the MLP head
        num_layers     : int
            Number of fully-connected layers in the MLP head
        dropout        : float
            Dropout probability
        use_lstm       : bool
            Whether to include an LSTM encoder for raw sequence data
        lstm_input_dim : int, optional
            Number of features per timestep for the LSTM input
        lstm_hidden    : int
            Hidden size of the LSTM
        lstm_layers    : int
            Number of stacked LSTM layers
        bidirectional  : bool
            Whether to use a bidirectional LSTM
        """
        super().__init__()

        # LSTM
        self.use_lstm = use_lstm
        if use_lstm:
            assert lstm_input_dim is not None, "lstm_input_dim must be provided when use_lstm=True"

            self.lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
                bidirectional=bidirectional
            )

            self.lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        else:
            self.lstm_out_dim = 0

        # DNN
        total_input_dim = input_dim + self.lstm_out_dim

        layers = [
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]

        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x_feats, x_seq=None):
        """
        Parameters
        ----------
        x_feats : Tensor
            [batch, input_dim] — static or engineered features
        x_seq : Tensor, optional
            [batch, seq_len, lstm_input_dim] — raw sequence input (only used if use_lstm=True)
        """
        if self.use_lstm:
            assert x_seq is not None, \
                "x_seq must be provided when use_lstm=True"

            _, (h, _) = self.lstm(x_seq)
            seq_embed = h[-1]  # [batch, lstm_hidden]
            combined = torch.cat([x_feats, seq_embed], dim=1)
        else:
            combined = x_feats

        out = self.regressor(combined)
        return out

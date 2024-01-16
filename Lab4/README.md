### ASRModel:
pBLSTM encoder + simple MLP Decoder

```
ASRModel(
  (encoder): Encoder(
    (embedding): Conv1d(28, 64, kernel_size=(3,), stride=(2,), padding=(1,))
    (pBLSTMs): Sequential(
      (0): pBLSTM(
        (blstm): LSTM(128, 64, bidirectional=True)
      )
      (1): LockedDropout()
      (2): pBLSTM(
        (blstm): LSTM(256, 64, bidirectional=True)
      )
      (3): LockedDropout()
      (4): pBLSTM(
        (blstm): LSTM(256, 64, bidirectional=True)
      )
    )
  )
  (decoder): MLPDecoder(
    (fc1): Linear(in_features=128, out_features=64, bias=True)
    (bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=64, out_features=41, bias=True)
    (softmax): LogSoftmax(dim=2)
  )
```

### LASModel:
pBLSTM encoder + Attention Decoder
import torch
import matplotlib.pyplot as plt


pickle = torch.load('d_long.pkl', map_location='cpu')
loss = [i['loss_ae'] + i['loss_fm'] for i in pickle['dynamics']]
loss_ae = [i['loss_ae'] for i in pickle['dynamics']]
loss_fm = [i['loss_fm'] for i in pickle['dynamics']]
plt.plot(loss, label='loss')
plt.plot(loss_ae, label='ae')
plt.plot(loss_fm, label='fm')
plt.legend()
plt.show()

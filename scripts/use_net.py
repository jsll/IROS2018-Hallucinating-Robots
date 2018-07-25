import nets as hn
import data as hd

# Parameters
inf_replacement_n = 30
nan_replacement_n = 30
in_trim_sx = 199
in_trim_dx = 325
test_in_trim_sx = 210
test_in_trim_dx = 336
out_trim_sx = 20
out_trim_dx = 635
trim_length = 126

use_cuda = True

# Transformations
trans_in = transforms.Compose([hd.Interval(in_trim_sx, in_trim_dx),
                               hd.TrimToLength(trim_length),
                               hd.ReplaceInf(inf_replacement_n),
                               hd.ReplaceNan(nan_replacement_n),
                               hd.Normalize(0,30)])

net_layer_sizes = [16,32,64,128]
net_activation_gamma = 0.25
noise = 0.0

net = hn.to_cuda(hn.AutoEnc(net_layer_sizes, net_activation_gamma), use_cuda)

#Pre-trained network loading
net_load_file = 'Net-4-n-4_0--18_02_11_032538.pt'

print("Loading network parameters from file: %s" % net_load_file)
net.load_state_dict(tc.load("%s" % (net_load_file)))
print("Network loaded correctly")
net.eval()

if augmentations:
    data = augmentations(data)

# get the inputs
inputs = data['laser'].unsqueeze_(1).float()
truths = data['truth'].unsqueeze_(1).float()

# wrap them in Variable
inputs, truths = to_variable(inputs), to_variable(truths)

#network pass
outputs = net(inputs)
import torch


def get_mean_std(loader):
    """ to find the mean and std of data """
    channel_sum, channel_squared_sum, num_batches = 0,0,0

    for data, _ in loader: 
        channel_sum += torch.mean(data,dim=[0,2,3]) # data => [batch, channels, height, width]
        channel_squared_sum += torch.mean(data**2, dim=[0,2,3] )
        num_batches +=1

    mean = channel_sum/num_batches
    std = (channel_squared_sum/num_batches-mean**2)**0.5

    return mean,std
    

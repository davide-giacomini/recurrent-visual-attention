import time
import torch
from tqdm import tqdm

import modules
from utils import AverageMeter


class LookupOnlyInference:
    """A size-agnostic inference model [1].

    LOI is an inference methodology which harnesses a look-up
    table instead of normal computations of the netowkr.
    More information in the reference.

    References:
      [1]: Giacomini et. al., TODO: link to the paper
    """

    def __init__(self, config, data_loader):
        """Constructor.

        Args:
          config: configuration file with hyperparameters.
          k: number of patches to extract per glimpse.
          s: scaling factor that controls the size of successive patches.
          c: number of channels in each image.
          h_g: hidden layer size of the fc layer for `phi`. (config.loc_hidden)
          h_l: hidden layer size of the fc layer for `l`. (config.glimpse_hidden)
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
          num_classes: number of classes in the dataset.
          num_glimpses: number of glimpses to take per image,
            i.e. number of BPTT steps.
        """

        self.clustered_tree_path = config.clustered_tree_path
        self.data_loader = data_loader
        self.ckpt_dir = config.ckpt_dir
        
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
    def test(self):
        """Testing actual implementation of LOI with testing dataset
        """

        patch_size, num_patches, glimpse_scale, quant_bits_phi, model_name = self.retrieve_from_file(self.ckpt_dir)

        batch_time = AverageMeter()
        tic = time.time()
        
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.data_loader):
                x, y = x.to(self.device), y.to(self.device)

                # duplicate M times
                x = x.repeat(self.M, 1, 1, 1)

                retina = modules.Retina(patch_size, num_patches, glimpse_scale, quant_bits_phi, self.ckpt_dir, model_name)

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset() # (B,64)
                phi = retina.foveate(x, l_t) # (B,48)
                
                for t in range(self.num_glimpses - 1):
                    # closest_outputs = compute_closest_outputs(df, self.config, starting_cluster_dir, phi, h_t, l_t, self.device, a, b, c)   # (B,M-114)
                    closest_outputs = compute_closest_outputs(h_t, l_t, phi, a, b, c) # (B, M-114)

                    h_t = closest_outputs[:, :self.config.output_size_ht]
                    l_t = closest_outputs[:, self.config.output_size_ht:self.config.output_size_ht+2].long()
                    phi = retina.foveate(x, l_t, location_normalized=True)

                # closest_outputs = compute_closest_outputs(df, self.config, starting_cluster_dir, phi, h_t, l_t, self.device, a, b, c)   # (B,M-114)
                closest_outputs = compute_closest_outputs(h_t, l_t, phi, a, b, c) # (B, M-114)

                pred = closest_outputs[:, self.config.output_size_ht+2]
                
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()
                acc = 100 * (correct.sum() / len(y))

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)
                pbar.set_description(
                        (
                            "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                                (toc - tic), acc.item()
                            )
                        )
                    )
                pbar.update(self.batch_size)

        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc

        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
                correct, self.num_test, perc, error
            )
        )

        return perc
    
    def retrieve_from_file(self):
        pass
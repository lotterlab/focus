# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
# Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
# Implementation based on PyTorch ResNet implementation: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import math
import torch
import torch.nn as nn
import pdb
import numpy as np
from onconet.models.pools.factory import get_pool
from onconet.models.spatial_transformers.factory import get_spatial_transformer
from onconet.models.cumulative_probability_layer import Cumulative_Probability_Layer

import os
import matplotlib.pyplot as plt
from PIL import Image

class ResNet(nn.Module):
    """
        A ResNet model. Blocks can be Basic, Non-local, bottleneck or
        anything in onconet.models.blocks and intermixed in any order.
        This is a slight generalization of orginal resnet model,
        which assumed a homogenous block type.
    """

    def __init__(self, layers, args):
        """Initializes a generalized resnet. Supports arbitrary block configurations per layer.

        Arguments:
            layers(list): A length-4 list with the list of block
            classes in each of the 4 layers. Blocks can be
            basic blocks, bottlenecks, non-locals etc.

            num_classes(int): The number of classes the network
                is predicting between, i.e. the size of the final
                layer of the network.
            args(Args): configuration of experiment. Used to determine num gpus,
            cuda mode, etc.
        """

        super(ResNet, self).__init__()

        self.args = args
        self.args.wrap_model = False

        if hasattr(args, 'use_spatial_transformer') and args.use_spatial_transformer:
            self.stn = get_spatial_transformer(args.spatial_transformer_name)(args)

        self.args.hidden_dim = 512 * args.block_widening_factor
        input_dim = self.args.input_dim if self.args.use_precomputed_hiddens else self.args.num_chan
        self.inplanes = max(64 * args.block_widening_factor, input_dim)


        self.all_blocks = []
        if not self.args.use_precomputed_hiddens:
            downsampler = Downsampler(self.inplanes, input_dim)
            self.add_module('downsampler', downsampler)
            self.all_blocks.append('downsampler')

        layer_modules = [(self._make_layer(self.inplanes, layers[0]), 'layer1_{}')]
        current_dim = self.inplanes
        indx = 1
        for layer_i in layers[1:]:
            indx += 1
            current_dim = min(current_dim * 2, 1024)
            layer_modules.append(
                            (self._make_layer(current_dim, layer_i, stride=2),
                             'layer{}_'.format(indx)+'{}')
                            )
        args.hidden_dim = current_dim

        '''
            For all layers, register all constituent blocks to the module,
            and record block names for later access in self.all_blocks
        '''
        for layer, layer_name in layer_modules:
            for indx, block in enumerate(layer):
                block_name = layer_name.format(indx)
                self.add_module(block_name, block)
                self.all_blocks.append(block_name)

        last_block = layers[-1][-1]

        pool_name = args.pool_name
        if args.use_risk_factors:
            pool_name = 'DeepRiskFactorPool' if self.args.deep_risk_factor_pool else 'RiskFactorPool'
        self.pool = get_pool(pool_name)(args, args.hidden_dim)

        if not self.pool.replaces_fc():
            # Cannot not placed on self.all_blocks since requires intermediate op
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=args.dropout)
            self.fc = nn.Linear(args.hidden_dim, args.num_classes)

        if args.use_region_annotation and args.region_annotation_loss_type == 'pred_region':
            self.region_fc = nn.Conv2d(current_dim, 1, kernel_size=args.region_annotation_pred_kernel_size, padding=(args.region_annotation_pred_kernel_size -1) // 2)

        if args.predict_birads:
            self.birads_fc =  nn.Linear(args.hidden_dim, 2)

        if args.survival_analysis_setup:
            self.prob_of_failure_layer = Cumulative_Probability_Layer(args.hidden_dim, args, max_followup=args.max_followup)

        self.gpu_to_layer_assignments = self.get_gpu_to_layer()

    def get_gpu_to_layer(self):
        '''
            Given args.model_parallel, args.num_shards, will try to best
            balance layers across gpus given the number of gpus.

            returns:
            -gpu_to_layers: a list of lists of length num_shards. Each interior
            list consist of layer names to place on that index's gpu.
        '''
        if self.args.model_parallel and self.args.num_shards > 1:
            num_shards = self.args.num_shards
        else:
            num_shards = 1

        gpu_to_layers = np.array_split(self.all_blocks, num_shards)
        self._validate_gpu_assignments(gpu_to_layers)
        return gpu_to_layers

    def _validate_gpu_assignments(self, gpu_to_layers):
        """Confirms that all layers from self.all_blocks are in gpu_to_layers.

        Arguments:
            gpu_to_layers: A list of lists containing the layers assigned to each GPU.

        Raises:
            Exception if the the layers in self.all_blocks and the layers in gpu_to_layers
            don't match.
        """

        original_layers = set(self.all_blocks)
        layers_assigned = set([layer for layers in gpu_to_layers for layer in layers])

        if original_layers != layers_assigned:
            extra_layers = layers_assigned - original_layers
            missing_layers = original_layers - layers_assigned

            raise Exception(
                'GPU partitioned layers don\'t match original layers.\n\t{}\n\t{}'.format(
                    'Extra layers: {}'.format(extra_layers) if len(extra_layers) > 0 else '',
                    'Missing layers: {}'.format(missing_layers) if len(missing_layers) > 0 else ''
                )
            )

    def _make_layer(self, planes, blocks, stride=1):
        """Builds a layer of the ResNet.

        Arguments:
            planes(int): The number of filters to use in convolutions
                and therefore the depth (number of channels) of the output.
            blocks: list of block classes for this layer.
            stride(int): The stride to use in the convolutionals.

        Returns:
            A Sequential model containing all the blocks in this layer.
        """
        layers = []

        for i, block in enumerate(blocks):
            if (i == 0 and stride != 1) or self.inplanes != planes * block.expansion:

                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )
            else:
                downsample = None

            if i != 0:
                stride = 1

            layers.append(block(self.args,
                                self.inplanes,
                                planes,
                                stride=stride,
                                downsample=downsample
                                ))

            self.inplanes = planes * block.expansion

        return layers

    def forward(self, x, risk_factors=None, batch=None,
                CC_index=4, y_cc_l=4, y_cc_u=4, x_cc_l=4, x_cc_u=4,
                MLO_index=4, y_mlo_l=4, y_mlo_u=4, x_mlo_l=4, x_mlo_u=4):
        """Computes a forward pass of the model.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            The result of feeding the input through the model.
        """

        # Go through all layers up to fc
        if self.args.use_precomputed_hiddens:
            x = x.transpose(2,1)
        if hasattr(self.args, 'use_spatial_transformer') and self.args.use_spatial_transformer:
            x = self.stn(x)
        for gpu, layers in enumerate(self.gpu_to_layer_assignments):
            if self.args.cuda and self.args.model_parallel:
                x = x.cuda(gpu)

            print("Input to ResNet:", x.size())
            input_exam=x.clone()

            img_mean = 7047.99
            img_std = 12005.50
            denormalized_images=input_exam*img_std + img_mean
            denormalized_images=denormalized_images.cpu()
            grayscale_images=denormalized_images[:, 0, :, :]
            for i in range(4):
                image_np = grayscale_images[i].numpy()
                # Normalize to [0, 1] range
                image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
                # Convert to 8-bit unsigned integer
                image_np = (image_np * 255).astype(np.uint8)
                print(image_np.shape)
                image_pil = Image.fromarray(image_np, mode='L')
                if (i==0):
                    image_pil.save(os.path.join('/root/OncoNet', f'R_CC.png'))
                elif (i==1):
                    image_pil.save(os.path.join('/root/OncoNet', f'R_MLO.png'))
                elif (i==2):
                    image_pil.save(os.path.join('/root/OncoNet', f'L_CC.png'))
                else:
                    image_pil.save(os.path.join('/root/OncoNet', f'L_MLO.png'))

            print("# downsampler")
            x = self.downsampler.conv1(x) # kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
            print(x.size())
            x = self.downsampler.bn1(x)
            x = self.downsampler.relu(x)
            x = self.downsampler.maxpool(x) # kernel_size=3, stride=2, padding=1, dilation=1
            print(x.size())

            conv_details=[]
            print("# layer1_0")
            identity = x.clone()
            out = self.layer1_0.conv1(x) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer1_0.conv1.kernel_size, 
                                 self.layer1_0.conv1.padding, 
                                 self.layer1_0.conv1.stride])
            print(out.size())
            out = self.layer1_0.bn1(out)
            out = self.layer1_0.relu(out)
            out = self.layer1_0.conv2(out) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer1_0.conv2.kernel_size, 
                                 self.layer1_0.conv2.padding, 
                                 self.layer1_0.conv2.stride])
            print(out.size())
            out = self.layer1_0.bn2(out)
            x = out + identity
            x = self.layer1_0.relu(x)
            print(x.size())

            print("# layer1_1")
            identity = x.clone()
            out = self.layer1_1.conv1(x) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer1_1.conv1.kernel_size, 
                                 self.layer1_1.conv1.padding, 
                                 self.layer1_1.conv1.stride])
            print(out.size())
            out = self.layer1_1.bn1(out)
            out = self.layer1_1.relu(out)
            out = self.layer1_1.conv2(out) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer1_1.conv2.kernel_size, 
                                 self.layer1_1.conv2.padding, 
                                 self.layer1_1.conv2.stride])
            print(out.size())
            out = self.layer1_1.bn2(out)
            x = out + identity
            x = self.layer1_1.relu(x)
            print(x.size())

            print("# layer2_0")
            identity = x.clone()
            identity = self.layer2_0.downsample(identity) # kernel_size=(1, 1), stride=(2, 2)
            out = self.layer2_0.conv1(x) # kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            conv_details.append([self.layer2_0.conv1.kernel_size, 
                                 self.layer2_0.conv1.padding, 
                                 self.layer2_0.conv1.stride])
            print(out.size())
            out = self.layer2_0.bn1(out)
            out = self.layer2_0.relu(out)
            out = self.layer2_0.conv2(out) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer2_0.conv2.kernel_size, 
                                 self.layer2_0.conv2.padding, 
                                 self.layer2_0.conv2.stride])
            print(out.size())
            out = self.layer2_0.bn2(out)
            x = out + identity
            x = self.layer2_0.relu(x)
            print(x.size())

            print("# layer2_1")
            identity=x.clone()
            out = self.layer2_1.conv1(x) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer2_1.conv1.kernel_size, 
                                 self.layer2_1.conv1.padding, 
                                 self.layer2_1.conv1.stride])
            print(out.size())
            out = self.layer2_1.bn1(out)
            out = self.layer2_1.relu(out)
            out = self.layer2_1.conv2(out) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer2_1.conv2.kernel_size, 
                                 self.layer2_1.conv2.padding, 
                                 self.layer2_1.conv2.stride])
            print(out.size())
            out = self.layer2_1.bn2(out)
            x = out + identity
            x = self.layer2_1.relu(x) 
            print(x.size())  

            print("# layer3_0")
            identity = x.clone()
            identity = self.layer3_0.downsample(identity) # kernel_size=(1, 1), stride=(2, 2)
            out = self.layer3_0.conv1(x) # kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            conv_details.append([self.layer3_0.conv1.kernel_size, 
                                 self.layer3_0.conv1.padding, 
                                 self.layer3_0.conv1.stride])
            print(out.size())
            out = self.layer3_0.bn1(out)
            out = self.layer3_0.relu(out)
            out = self.layer3_0.conv2(out) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer3_0.conv2.kernel_size, 
                                 self.layer3_0.conv2.padding, 
                                 self.layer3_0.conv2.stride])
            print(out.size())
            out = self.layer3_0.bn2(out)
            x = out + identity
            x = self.layer3_0.relu(x)
            print(x.size())

            print("# layer3_1")
            identity=x.clone()
            out = self.layer3_1.conv1(x) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer3_1.conv1.kernel_size, 
                                 self.layer3_1.conv1.padding, 
                                 self.layer3_1.conv1.stride])
            print(out.size())
            out = self.layer3_1.bn1(out)
            out = self.layer3_1.relu(out)
            out = self.layer3_1.conv2(out) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer3_1.conv2.kernel_size, 
                                 self.layer3_1.conv2.padding, 
                                 self.layer3_1.conv2.stride])
            print(out.size())
            out = self.layer3_1.bn2(out)
            x = out + identity
            x = self.layer3_1.relu(x) 
            print(x.size())

            print("# layer4_0")
            identity = x.clone()
            identity = self.layer4_0.downsample(identity) # kernel_size=(1, 1), stride=(2, 2)
            out = self.layer4_0.conv1(x) # kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            conv_details.append([self.layer4_0.conv1.kernel_size, 
                                 self.layer4_0.conv1.padding, 
                                 self.layer4_0.conv1.stride])
            print(out.size())
            out = self.layer4_0.bn1(out)
            out = self.layer4_0.relu(out)
            out = self.layer4_0.conv2(out) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer4_0.conv2.kernel_size, 
                                 self.layer4_0.conv2.padding, 
                                 self.layer4_0.conv2.stride])
            print(out.size())
            out = self.layer4_0.bn2(out)
            x = out + identity
            x = self.layer4_0.relu(x)
            print(x.size())

            print("# layer4_1")
            identity=x.clone()
            out = self.layer4_1.conv1(x) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer4_1.conv1.kernel_size, 
                                 self.layer4_1.conv1.padding, 
                                 self.layer4_1.conv1.stride])
            print(out.size())
            out = self.layer4_1.bn1(out)
            out = self.layer4_1.relu(out)
            out = self.layer4_1.conv2(out) # kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            conv_details.append([self.layer4_1.conv2.kernel_size, 
                                 self.layer4_1.conv2.padding, 
                                 self.layer4_1.conv2.stride])
            print(out.size())
            out = self.layer4_1.bn2(out)
            x = out + identity
            x = self.layer4_1.relu(x) 
            print(x.size())

            x[CC_index, :, y_cc_l:y_cc_u, x_cc_l:x_cc_u]=-np.inf 
            x[MLO_index, :, y_mlo_l:y_mlo_u, x_mlo_l:x_mlo_u]=-np.inf 

        # # print(conv_details)
        # max_locations=self.aggregate_and_classify(x, risk_factors=risk_factors)
        # # print("max_locations")
        # # print(max_locations[0, 0, :])

        # # Steps: 
        # # 1. For each image, for each channel, max_locations has a pair of coordinates (x, y) that correspond to output x (layer4_1.relu(x))
        # # 2. Use x and y coordinates to find the corresponding bottom left (kernel_index=0) and top right (kernel_index=2) coordinates that correspond to input of layer4_1.conv2
        # # 3. For the remaining layers (iterate from layer4_1.conv1 to layer1_0.conv1)
        # # 4. Take the bottom left and top right output coordinates and find the corresponding bottom left and top right input coordinates.
        
        # def corresponding_input_coordinates(output_coordinates, kernel_index, padding, stride):
        #     h_o=output_coordinates[0]
        #     w_o=output_coordinates[1]
        #     h_i=(h_o*stride) - padding + kernel_index
        #     w_i=(w_o*stride) - padding + kernel_index
        #     return torch.tensor((h_i, w_i))
        
        # print("layer4_1.conv2")
        # input_coordinates=torch.zeros((4, 512, 2, 2), dtype=torch.long)
        # for image in range(max_locations.size()[0]): # 4
        #     for channel in range(max_locations.size()[1]): # 512
        #         max_location = max_locations[image, channel, :]
        #         # print(max_location)
        #         # bottom left
        #         # input_coordinates[image, channel, 0, 0] = corresponding_input_coordinates(max_location, 0, 1, 1)[0]
        #         # input_coordinates[image, channel, 0, 1] = corresponding_input_coordinates(max_location, 0, 1, 1)[1]
        #         # input_coordinates[image, channel, 0, :] = corresponding_input_coordinates(max_location, 0, 1, 1)
        #         input_coordinates[image, channel, 0, :] = corresponding_input_coordinates(max_location, 0, 
        #                                                                                   conv_details[-1][1][0],
        #                                                                                   conv_details[-1][2][0])

        #         # top right
        #         # input_coordinates[image, channel, 1, 0] = corresponding_input_coordinates(max_location, 2, 1, 1)[0]
        #         # input_coordinates[image, channel, 1, 1] = corresponding_input_coordinates(max_location, 2, 1, 1)[1]
        #         # input_coordinates[image, channel, 1, :] = corresponding_input_coordinates(max_location, 2, 1, 1)
        #         input_coordinates[image, channel, 1, :] = corresponding_input_coordinates(max_location, 2, 
        #                                                                                   conv_details[-1][1][0],
        #                                                                                   conv_details[-1][2][0])
        #         # print(input_coordinates[0,0,:,:])
        
        # print(input_coordinates[0,0,0,:])
        # print(input_coordinates[0,0,1,:])

        # print("layer4_1.conv1 --> layer1_0.conv1")
        # for conv_layer in range(len(conv_details)-2, -1, -1):
        #     for image in range(max_locations.size()[0]): # 4
        #         for channel in range(max_locations.size()[1]): # 512
        #             bottom_left = input_coordinates[image,channel, 0, :]
        #             top_right = input_coordinates[image,channel, 1, :]
        #             input_coordinates[image,channel, 0, :]=corresponding_input_coordinates(bottom_left, 0, 
        #                                                                                    conv_details[conv_layer][1][0], 
        #                                                                                    conv_details[conv_layer][2][0])
        #             input_coordinates[image,channel, 1, :]=corresponding_input_coordinates(top_right, 2, 
        #                                                                                    conv_details[conv_layer][1][0], 
        #                                                                                    conv_details[conv_layer][2][0])

        # print(input_coordinates[0,0,0,:])
        # print(input_coordinates[0,0,1,:])

        # # downsampler.maxpool
        # # kernel_size=3, stride=2, padding=1, dilation=1
        # # in = (out * stride) - (padding) + (kernel_index * dilation)
        # print("downsampler.maxpool")
        # for image in range(max_locations.size()[0]): # 4
        #     for channel in range(max_locations.size()[1]): # 512
        #         input_coordinates[image,channel, 0, 0] = (input_coordinates[image,channel, 0, 0]*2) - 1 + (0*1)
        #         input_coordinates[image,channel, 0, 1] = (input_coordinates[image,channel, 0, 1]*2) - 1 + (0*1)
        #         input_coordinates[image,channel, 1, 0] = (input_coordinates[image,channel, 1, 0]*2) - 1 + (2*1)
        #         input_coordinates[image,channel, 1, 1] = (input_coordinates[image,channel, 1, 1]*2) - 1 + (2*1)

        # print(input_coordinates[0,0,0,:])
        # print(input_coordinates[0,0,1,:])

        # # downsampler.conv1
        # # kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        # # in = (out*stride) - padding + kernel_index
        # print("downsampler.conv1")
        # for image in range(max_locations.size()[0]): # 4
        #     for channel in range(max_locations.size()[1]): # 512
        #         bottom_left = input_coordinates[image,channel, 0, :]
        #         top_right = input_coordinates[image,channel, 1, :]
        #         input_coordinates[image,channel, 0, :]=corresponding_input_coordinates(bottom_left, 0, 3, 2)
        #         input_coordinates[image,channel, 1, :]=corresponding_input_coordinates(top_right, 6, 3, 2)
                
        # print(input_coordinates[0,0,0,:])
        # print(input_coordinates[0,0,1,:])

        # # Heatmap Overlay
        # heatmaps=torch.zeros((4, 2048, 1664))
        # for image in range(max_locations.size()[0]): # 4
        #     for channel in range(max_locations.size()[1]): # 512
        #         bottom_left = input_coordinates[image, channel, 0]
        #         top_right = input_coordinates[image, channel, 1]

        #         # Coordinates of interest
        #         y1, x1 = bottom_left
        #         y2, x2 = top_right

        #         heatmaps[image, y1:y2+1, x1:x2+1] += 1

        # # normalized_heatmap = heatmaps/heatmaps.max(dim=(1, 2), keepdim=True)[0]
        # # Normalize the heatmaps by finding the maximum value over each image
        # heatmaps_max = heatmaps.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        # # Prevent division by zero by ensuring heatmaps_max is not zero
        # heatmaps_max[heatmaps_max == 0] = 1
        # # Now normalize the heatmap
        # normalized_heatmap = heatmaps / heatmaps_max
        
        # for image in range(max_locations.size()[0]): # 4
        #     plt.figure(figsize=[10, 10])
        #     input_image = input_exam[image].permute(1, 2, 0).cpu().numpy()
        #     heatmap = normalized_heatmap[image].cpu().numpy()
        #     plt.imshow(input_image)
        #     plt.imshow(heatmap, cmap='jet', alpha=0.5)
        #     plt.title(f'Heatmap for Image {image}')
        #     plt.axis('off')
        #     plt.savefig(os.path.join('/root/OncoNet', f'heatmap_image_{image}.png'))
        #     plt.close()

        # Original Code
            # for name in layers:
            #     layer = self._modules[name]
            #     x = layer(x)
        logit, hidden, max_locations = self.aggregate_and_classify(x, risk_factors=risk_factors)
        # print(conv_details)
        # max_locations=self.aggregate_and_classify(x, risk_factors=risk_factors)
        # print("max_locations")
        # print(max_locations[0, 0, :])

        # Steps: 
        # 1. For each image, for each channel, max_locations has a pair of coordinates (x, y) that correspond to output x (layer4_1.relu(x))
        # 2. Use x and y coordinates to find the corresponding bottom left (kernel_index=0) and top right (kernel_index=2) coordinates that correspond to input of layer4_1.conv2
        # 3. For the remaining layers (iterate from layer4_1.conv1 to layer1_0.conv1)
        # 4. Take the bottom left and top right output coordinates and find the corresponding bottom left and top right input coordinates.
        
        def corresponding_input_coordinates(output_coordinates, kernel_index, padding, stride):
            h_o=output_coordinates[0]
            w_o=output_coordinates[1]
            h_i=(h_o*stride) - padding + kernel_index
            w_i=(w_o*stride) - padding + kernel_index
            return torch.tensor((h_i, w_i))
        
        print(max_locations[0, 0, :])
        print("layer4_1.conv2")
        input_coordinates=torch.zeros((4, 512, 2, 2), dtype=torch.long)
        for image in range(max_locations.size()[0]): # 4
            for channel in range(max_locations.size()[1]): # 512
                max_location = max_locations[image, channel, :]
                # print(max_location)
                # bottom left
                # input_coordinates[image, channel, 0, 0] = corresponding_input_coordinates(max_location, 0, 1, 1)[0]
                # input_coordinates[image, channel, 0, 1] = corresponding_input_coordinates(max_location, 0, 1, 1)[1]
                # input_coordinates[image, channel, 0, :] = corresponding_input_coordinates(max_location, 0, 1, 1)
                input_coordinates[image, channel, 0, :] = corresponding_input_coordinates(max_location, 0, 
                                                                                          conv_details[-1][1][0],
                                                                                          conv_details[-1][2][0])
                # top right
                # input_coordinates[image, channel, 1, 0] = corresponding_input_coordinates(max_location, 2, 1, 1)[0]
                # input_coordinates[image, channel, 1, 1] = corresponding_input_coordinates(max_location, 2, 1, 1)[1]
                # input_coordinates[image, channel, 1, :] = corresponding_input_coordinates(max_location, 2, 1, 1)
                input_coordinates[image, channel, 1, :] = corresponding_input_coordinates(max_location, 2, 
                                                                                          conv_details[-1][1][0],
                                                                                          conv_details[-1][2][0])
        print(input_coordinates[0,0,0,:])
        print(input_coordinates[0,0,1,:])

        print("layer4_1.conv1 --> layer1_0.conv1")
        for conv_layer in range(len(conv_details)-2, -1, -1):
            for image in range(max_locations.size()[0]): # 4
                for channel in range(max_locations.size()[1]): # 512
                    bottom_left = input_coordinates[image,channel, 0, :]
                    top_right = input_coordinates[image,channel, 1, :]
                    input_coordinates[image,channel, 0, :]=corresponding_input_coordinates(bottom_left, 0, 
                                                                                           conv_details[conv_layer][1][0], 
                                                                                           conv_details[conv_layer][2][0])
                    input_coordinates[image,channel, 1, :]=corresponding_input_coordinates(top_right, 2, 
                                                                                           conv_details[conv_layer][1][0], 
                                                                                           conv_details[conv_layer][2][0])

        print(input_coordinates[0,0,0,:])
        print(input_coordinates[0,0,1,:])

        # downsampler.maxpool
        # kernel_size=3, stride=2, padding=1, dilation=1
        # in = (out * stride) - (padding) + (kernel_index * dilation)
        print("downsampler.maxpool")
        for image in range(max_locations.size()[0]): # 4
            for channel in range(max_locations.size()[1]): # 512
                input_coordinates[image,channel, 0, 0] = (input_coordinates[image,channel, 0, 0]*2) - 1 + (0*1)
                input_coordinates[image,channel, 0, 1] = (input_coordinates[image,channel, 0, 1]*2) - 1 + (0*1)
                input_coordinates[image,channel, 1, 0] = (input_coordinates[image,channel, 1, 0]*2) - 1 + (2*1)
                input_coordinates[image,channel, 1, 1] = (input_coordinates[image,channel, 1, 1]*2) - 1 + (2*1)

        print(input_coordinates[0,0,0,:])
        print(input_coordinates[0,0,1,:])

        # downsampler.conv1
        # kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        # in = (out*stride) - padding + kernel_index
        print("downsampler.conv1")
        for image in range(max_locations.size()[0]): # 4
            for channel in range(max_locations.size()[1]): # 512
                bottom_left = input_coordinates[image,channel, 0, :]
                top_right = input_coordinates[image,channel, 1, :]
                input_coordinates[image,channel, 0, :]=corresponding_input_coordinates(bottom_left, 0, 3, 2)
                input_coordinates[image,channel, 1, :]=corresponding_input_coordinates(top_right, 6, 3, 2)
                
        print(input_coordinates[0,0,0,:])
        print(input_coordinates[0,0,1,:])

        # Heatmap Overlay
        heatmaps=torch.zeros((4, 2048, 1664))
        for image in range(max_locations.size()[0]): # 4
            for channel in range(max_locations.size()[1]): # 512
                bottom_left = input_coordinates[image, channel, 0]
                top_right = input_coordinates[image, channel, 1]

                # Coordinates of interest
                y1, x1 = bottom_left
                y2, x2 = top_right

                heatmaps[image, x1:x2+1, y1:y2+1] += 1

        # Save heatmaps as array
        np.save('/root/OncoNet/heatmap_array.npy', heatmaps)
        
        # normalized_heatmap = heatmaps/heatmaps.max(dim=(1, 2), keepdim=True)[0]
        # Normalize the heatmaps by finding the maximum value over each image
        heatmaps_max = heatmaps.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        # Prevent division by zero by ensuring heatmaps_max is not zero
        heatmaps_max[heatmaps_max == 0] = 1
        # Now normalize the heatmap
        normalized_heatmap = heatmaps / heatmaps_max
        
        for image in range(max_locations.size()[0]): # 4
            plt.figure(figsize=[10, 10])
            input_image = input_exam[image].permute(1, 2, 0).cpu().numpy()
            heatmap = normalized_heatmap[image].cpu().numpy()
            plt.imshow(input_image)
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.title(f'Heatmap for Image {image}')
            plt.axis('off')
            # plt.savefig(os.path.join('/root/OncoNet', f'heatmap_image_{image}.png'))
            if (image==0):
                plt.savefig(os.path.join('/root/OncoNet', f'heatmap_R_CC.png'))
            elif (image==1):
                plt.savefig(os.path.join('/root/OncoNet', f'heatmap_R_MLO.png'))
            elif (image==2):
                plt.savefig(os.path.join('/root/OncoNet', f'heatmap_L_CC.png'))
            elif (image==3):
                plt.savefig(os.path.join('/root/OncoNet', f'heatmap_L_MLO.png'))
            plt.close()

        activ_dict = {'activ':x}
        if self.args.use_region_annotation:
            activ_dict['region_logit'] = self.region_fc(x)
        if self.args.predict_birads:
            activ_dict['birads_logit'] = self.birads_fc(hidden)

        if self.args.pred_risk_factors:
            try:
                activ_dict['pred_rf_loss'] = self.pool.get_pred_rf_loss(hidden, risk_factors)
            except:
                pass
        if self.args.use_precomputed_hiddens:
            return logit, logit, logit, hidden
        else:
            return logit, hidden, activ_dict


    def aggregate_and_classify(self, x, risk_factors=None):
        # if self.args.use_risk_factors:
        #     max_locations=self.pool(x, risk_factors)
        # return max_locations

        # Pooling layer
        if self.args.use_risk_factors:
            logit, hidden, max_locations = self.pool(x, risk_factors)
        else:
            logit, hidden = self.pool(x)

        if not self.pool.replaces_fc():
            # self.fc is always on last gpu, so direct call of fc(x) is safe
            try:
                # placed in try catch for back compatbility.
                hidden = self.relu(hidden)
            except :
                pass
            hidden = self.dropout(hidden)
            logit = self.fc(hidden)

        if self.args.survival_analysis_setup:
            logit = self.prob_of_failure_layer(hidden)
        return logit, hidden, max_locations



    def cuda(self, device=None):
        '''
            Moves all submodules to gpu according to gpu_to_layer_assignments.
            , and returns model.
            Does not currently support start device different from 0.
            self.fc is always placed on the last GPU to reduce the amount of cross GPU communication.

            Note, must be called directly from parent module must overide it's .cuda() function to directly call this .cuda() fn trigger this
            method. Generic .cuda() call skips this function, and recurses to leaf nodes directly.
        '''
        if not self.args.model_parallel:
             return self._apply(lambda t: t.cuda(device))

        for gpu, layers in enumerate(self.gpu_to_layer_assignments):
            # fetch layers for GPU at device gpu(int)
            for name in layers:
                # move each layer (identified by module name) to corresponding gpu
                self._modules[name] = self._modules[name].cuda(gpu)

        if not self.pool.replaces_fc():
            # place fc layer at last gpu, so no need for extra gpu communication
            self.fc.cuda(len(self.gpu_to_layer_assignments) - 1)

        return self



class Downsampler(nn.Module):
    """Downsampling layers for ResNet. Downsamples input by 4x"""


    def __init__(self, inplanes, num_chan=3):

        self.inplanes = inplanes
        super(Downsampler, self).__init__()
        self.conv1 = nn.Conv2d(num_chan, inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
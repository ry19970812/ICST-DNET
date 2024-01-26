# STCL module
#-----------------------------------------------------------------------------
class residual_first(nn.Module):
    def __init__(self):
        super(residual_first, self).__init__()
        self.linear_alignment = nn.Linear(l_dimension, P_dimension, bias=False)
        self.inter_coff = nn.Parameter(torch.randn(l_dimension, P_dimension))
        self.layer_weight = nn.Parameter(torch.FloatTensor([1]))

    def forward(self, X_input):
        X = torch.transpose(X_input, 1, 2)
        X_id = torch.transpose(X_input, 1, 2)
        # print(X.shape) # shape torch.Size([20, 2, 6])
        X = torch.matmul(X, self.inter_coff)
        # X = X * self.inter_coff
        # print(X.shape) # shape torch.Size([20, 2, 32]) (batch_size, m, p)
        identity_mapping = self.linear_alignment(X_id)
        # print(identity_mapping.shape) # shape torch.Size([20, 2, 32])
        X = F.relu(self.layer_weight * X + identity_mapping)
        # print(X.shape) # shape torch.Size([20, 228, 32])
        return X, self.layer_weight


class residual_normal(nn.Module):
    def __init__(self):
        super(residual_normal, self).__init__()
        self.resiidual_coff = nn.Parameter(torch.randn(P_dimension, P_dimension))
        self.second_layer_weight = nn.Parameter(torch.FloatTensor([Initial_value_second_layer]))
        self.third_layer_weight = nn.Parameter(torch.FloatTensor([Initial_value_third_layer]))
        # self.layer_weight = nn.Parameter(torch.FloatTensor([1]))
    def forward(self, residual_input, counts):
        # residual_output = torch.matmul(residual_input, self.resiidual_coff)
        # residual_output = F.relu(self.layer_weight * residual_output + residual_input)
        if counts == 1:
            residual_output = torch.matmul(residual_input, self.resiidual_coff)
            residual_output = F.relu(self.second_layer_weight * residual_output + residual_input)
            layer_weight = self.second_layer_weight
        if counts == 2:
            residual_output = torch.matmul(residual_input, self.resiidual_coff)
            residual_output = F.relu(self.third_layer_weight * residual_output + residual_input)
            layer_weight = self.third_layer_weight

        return residual_output, layer_weight
        # return residual_output, self.layer_weight



class module_one(nn.Module):
    def __init__(self):
        super(module_one, self).__init__()
        self.residual_special = residual_first()
        self.residual_normal = residual_normal()

    def forward(self, X_input):

        residual_result = []
        beta_layer_weight = []
        for i in range(num_residual):
            if i == 0:
                residual_special_output, beta_layer_one = self.residual_special(X_input)
                # print(residual_special_output.shape) # shape torch.Size([20, 2, 32])
                residual_result.append(residual_special_output)
                beta_layer_weight.append(beta_layer_one)
                # print(beta_layer_one)
            else:
                counts = i
                residual_input = residual_result[i-1]
                residual_output, beta_layer_others = self.residual_normal(residual_input, counts)
                # print(residual_output.shape)
                # break
                residual_result.append(residual_output)
                beta_layer_weight.append(beta_layer_others)
                # print(beta_layer_others)

        # print(len(residual_result)) # num_residual
        # print(len(beta_layer_weight)) # num_residual
        return residual_result, beta_layer_weight


class module_two(nn.Module):
    def __init__(self):
        super(module_two, self).__init__()
        self.casual_graph = nn.Parameter(torch.randn(num_sensors, num_sensors))

    def forward(self, residual_result, beta_layer_weight):
        # print(len(residual_result)) # num_residual
        # print(len(beta_layer_weight)) # num_residual
        module_two_output = []
        casual_weight = []
        for i in range(num_residual):
            casual_input = residual_result[i]
            casual_input = torch.transpose(casual_input, 1, 2)
            casual_output = torch.matmul(casual_input, self.casual_graph)
            # print(casual_output.shape) # shape torch.Size([20, 32, 228])
            casual_output = torch.transpose(casual_output, 1, 2)
            module_two_output.append(casual_output)
            casual_weight.append(self.casual_graph)
            # break


        # 因果图计算
        casual_final_output = torch.zeros([num_sensors, num_sensors]).to(device)
        for j in range(num_residual):
            layer_weight = beta_layer_weight[j]
            casual_weight_layer = casual_weight[j]
            casual_output = layer_weight * casual_weight_layer
            casual_final_output = casual_output + casual_final_output

        # print(casual_final_output.shape)
        return module_two_output, casual_final_output


class module_three(nn.Module):
    def __init__(self):
        super(module_three, self).__init__()
        self.non_linear = nn.Sequential(
            nn.Linear(P_dimension * num_residual, nonlinear_dimension),
            nn.ReLU(),
            nn.Linear(nonlinear_dimension, nonlinear_dimension)
        )

    def forward(self, module_two_output):
        for i in range(num_residual):
            if i == 0:
                a = module_two_output[i]
            else:
                b = module_two_output[i]
                c = torch.cat([a, b], dim=2)
                a = c
        # print(c.shape) # shape torch.Size([20, 228, 160]) (batch_size, timesteps, num_residual * one_residual_feature)
        module_three_output = self.non_linear(c)
        # print(module_three_output.shape) # shape torch.Size([20, 2, 256]) (batch_size, timesteps, nonlinear_dimension)
        return module_three_output


class diffusion_summation(nn.Module):
    def __init__(self):
        super(diffusion_summation, self).__init__()

    def forward(self, module_three_output):

        # for single_output in module_three_output:
        #     diffusion = single_output
        #     diffusion_sum =
        diffusion_sum = sum(module_three_output)
        return diffusion_sum



class traffic_diffusion(nn.Module):
    def __init__(self):
        super(traffic_diffusion, self).__init__()
        self.module_first = module_one()
        self.module_second = module_two()
        self.module_third = module_three()
        self.diffusion = diffusion_summation()
        # self.module_fourth = module_four()

    def forward(self, input_X):
        # 扩散相加
        diffusion_summation = []
        residual_result, beta_layer_weight = self.module_first(input_X)
        module_two_output, Casual_Graph = self.module_second(residual_result, beta_layer_weight)
        module_three_output = self.module_third(module_two_output)
        diffusion_summation.append(module_three_output)
        diffusion = self.diffusion(diffusion_summation)
        # print(diffusion.shape)
        # diffusion = diffusion.view(-1, num_sensors * nonlinear_dimension)
        # print(diffusion.shape) # shape torch.Size([20, 2 * 32])

        # final_output = self.module_fourth(module_three_output)
        # print(Casual_Graph.shape)
        # print(beta_layer_weight)
        return diffusion, Casual_Graph, beta_layer_weight


class STCL_module(nn.Module):
    def __init__(self):
        super(STCL_module, self).__init__()
        self.Casual_Diffusion = traffic_diffusion()
        self.regression = nn.Linear(P_dimension, 1)
        self.predicted_weight = nn.Parameter(torch.randn(predicted_timestep, nonlinear_dimension, 1))
        self.STEmbedding = STEmbedding()

        self.projection_STE = nn.Linear(D, 1)
    def forward(self, X_input_CD, X_TE_input, X_SE_input):
        # print(X_input_CD.shape) # shape torch.Size([128, 5, 207])
        diffusion, Casual_Graph, beta_layer_weight = self.Casual_Diffusion(X_input_CD)
        # print(diffusion.shape)
        diffusion = torch.unsqueeze(diffusion, dim=1)
        # print(diffusion.shape) # shape torch.Size([128, 1, 207, 32])
        diffusion_output = torch.matmul(diffusion, self.predicted_weight)
        # print(diffusion_output.shape) # shape torch.Size([128, 2, 207, 1])
        # diffusion_output = diffusion_output.view(-1, diffusion_output.shape[1], diffusion_output.shape[2] * diffusion_output.shape[3])
        # diffusion_output = self.regression(diffusion)
        # diffusion_output = diffusion_output.view(-1, diffusion_output.shape[1] * diffusion_output.shape[2])
        # print(diffusion_output.shape) # shape torch.Size([128, 207])


        # STE embedding
        # print(X_TE_input.shape)
        # print(X_SE_input.shape)
        X_TE_input = X_TE_input.contiguous().view(-1, history_timestep + predicted_timestep, 2)
        # print(X_TE_input.shape) # shape torch.Size([128, 7, 2])
        STE = self.STEmbedding(X_SE_input, X_TE_input)
        # print(STE.shape)
        His_STE = STE[:, :history_timestep, :, :]
        # print(His_STE.shape) # shape torch.Size([128, 5, 207, 8])
        Pred_STE = STE[:, history_timestep:, :, :]
        # print(Pred_STE.shape) # shape torch.Size([128, 2, 207, 8])

        # sum
        Pred_STE = self.projection_STE(Pred_STE)
        # print(Pred_STE.shape)
        Pred_STE_and_diffusion_output = Pred_STE + diffusion_output
        # print(Pred_STE_and_diffusion_output.shape)
        # Pred_STE_and_diffusion_output = Pred_STE_and_diffusion_output.view(-1, Pred_STE_and_diffusion_output.shape[1], Pred_STE_and_diffusion_output.shape[2] * Pred_STE_and_diffusion_output.shape[3])

        return Pred_STE_and_diffusion_output


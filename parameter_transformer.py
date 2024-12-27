import json
import os
import numpy as np
from eagent.config import cfg_dict


class Transform():
    def __init__(self, log_old, parameter_filename, cfg_name, output_dir):
        self.log_old = log_old
        self.parameter_filename = parameter_filename
        self.cfg_name = cfg_name
        self.output_dir = output_dir
        with open(os.path.join(log_old, "cfg.json"), "r") as f:
            self.cfg_old = json.load(f)
        with open(os.path.join(log_old, f"parameter_{parameter_filename}.json"), "r") as f:
            self.parameter_old = json.load(f)
        self.cfg_new = cfg_dict[cfg_name]
        with open(self.cfg_new["initial_params_filename"], "r") as f:
            self.parameter_new_format = json.load(f)
        self.max_num_limbs_old = self.cfg_old["max_num_limbs"]
        self.max_num_limbs_new = self.cfg_new["max_num_limbs"]
        self.survived_nodes = []

    def transform_structure_para(self, id, current, parent):
        for child in self.structure_tree[current]:
            self.survived_nodes.append(child)
            dof = int(self.dofs[child])
            self.structure_edges_new.append([parent, id, dof])
            parent_id = id
            self.mu_format_list[id] = self.mu_old_list[child]
            self.sigma_format_list[id] = self.sigma_old_list[child]
            id += 1
            # assert id < self.max_num_limbs_new
            if id >= self.max_num_limbs_new:
                raise Exception("########num of limbs is out of code graph#######")
            id = self.transform_structure_para(id, child, parent_id)
        return id
    
    def reshape_first_and_final_weights(self, net_arch_pi, net_arch_vf, num_obs, num_act, policy):
        pi_l1 = net_arch_pi[0]
        pi_l2 = net_arch_pi[1]
        vf_l1 = net_arch_vf[0]
        vf_l2 = net_arch_vf[1]
        # first layer: pi
        idx = num_obs * pi_l1
        pi_weights1 = policy[:idx].reshape(pi_l1, num_obs)
        pi_bias1 = policy[idx:idx+pi_l1]
        idx += pi_l1
        # second layer: pi
        pi_weights2 = policy[idx:idx+pi_l1*pi_l2]
        idx += pi_l1*pi_l2
        pi_bias2 = policy[idx:idx+pi_l2]
        idx += pi_l2
        # first layer: vf
        vf_weights1 = policy[idx:idx+num_obs*vf_l1].reshape(vf_l1, num_obs)
        idx += num_obs*vf_l1
        vf_bias1 = policy[idx:idx+vf_l1]
        idx += vf_l1
        # second layer: vf
        vf_weights2 = policy[idx:idx+vf_l1*vf_l2]
        idx += vf_l1*vf_l2
        vf_bias2 = policy[idx:idx+vf_l2]
        idx += vf_l2
        # third layer: pi
        pi_weights3 = policy[idx:idx+pi_l2*num_act].reshape(num_act, pi_l2)
        idx += pi_l2*num_act
        pi_bias3 = policy[idx:idx+num_act]
        idx += num_act
        # third layer: vf
        vf_weights3 = policy[idx:idx+vf_l2].reshape(1, vf_l2)
        idx += vf_l2
        vf_bias3 = policy[idx]
        
        assert idx==len(policy)-1
        
        return pi_weights1, pi_bias1, pi_weights2, pi_bias2, vf_weights1, vf_bias1, vf_weights2, vf_bias2, pi_weights3, pi_bias3, vf_weights3, vf_bias3

    def make_policy_para(self):
        cfg = self.cfg_new
        max_num_limbs = cfg["max_num_limbs"]
        pi_net = []
        vf_net = []
        num_obs = max_num_limbs * 4 + 11
        pi_net.append(num_obs)
        vf_net.append(num_obs)
        net_arch_pi = cfg["policy_kwargs"]["net_arch"][0]["pi"]
        net_arch_vf = cfg["policy_kwargs"]["net_arch"][0]["vf"]
        pi_net.extend(net_arch_pi)
        vf_net.extend(net_arch_vf)
        num_act = max_num_limbs * 2
        pi_net.append(num_act)
        vf_net.append(1)
        
        num_obs_old = self.max_num_limbs_old * 4 + 11
        num_act_old = self.max_num_limbs_old * 2
        
        # survived node: action
        self.survived_act_nodes = []
        for i in range(len(self.survived_nodes)):
            self.survived_act_nodes += [self.survived_nodes[i]*2, self.survived_nodes[i]*2 + 1]
        
        # survived node: observation
        self.survived_obs_nodes = []
        for i in range(len(self.survived_nodes)):
            self.survived_obs_nodes += [self.survived_nodes[i]*4, self.survived_nodes[i]*4 + 1, self.survived_nodes[i]*4 + 2, self.survived_nodes[i]*4 + 3]
        # self.survived_obs_nodes += np.arange(self.max_num_limbs_old*4, self.max_num_limbs_old*4 + 11).tolist()
        

        num_params_pi = 0
        for layer in range(len(pi_net) - 1):
            num_params_pi += pi_net[layer] * pi_net[layer + 1]
            num_params_pi += pi_net[layer + 1]

        num_params_vf = 0
        for layer in range(len(vf_net) - 1):
            num_params_vf += vf_net[layer] * vf_net[layer + 1]
            num_params_vf += vf_net[layer + 1]

        num_params = num_params_pi + num_params_vf
        policy_weights_new = (np.random.randn(num_params) * 0.1).tolist()
        action_net_bias = [0 for i in range(num_act)]
        policy_weights_new[- ((vf_net[-2] * vf_net[-1] + vf_net[-1]) + pi_net[-1]) : - (vf_net[-2] * vf_net[-1] + vf_net[-1])] = action_net_bias
        policy_weights_new[-1] = 0
        
        pw1, pb1, pw2, pb2, vw1, vb1, vw2, vb2, pw3, pb3, vw3, vb3 \
            = self.reshape_first_and_final_weights(net_arch_pi, net_arch_vf, num_obs, num_act, np.array(policy_weights_new))
        
        if self.max_num_limbs_old >= self.max_num_limbs_new:
            pi_weights1, pi_bias1, pi_weights2, pi_bias2,\
                vf_weights1, vf_bias1, vf_weights2, vf_bias2,\
                    pi_weights3, pi_bias3, vf_weights3, vf_bias3 \
                = self.reshape_first_and_final_weights(self.cfg_old['policy_kwargs']['net_arch'][0]['pi'],
                                                       self.cfg_old['policy_kwargs']['net_arch'][0]['vf'],
                                                       num_obs_old, num_act_old,
                                                       self.policy_old)
            pw1[:, :len(self.survived_obs_nodes)] = pi_weights1[:, self.survived_obs_nodes]
            pw1[:, -11:] = pi_weights1[:, -11:]
            pb1 = pi_bias1
            vw1[:, :len(self.survived_obs_nodes)] = vf_weights1[:, self.survived_obs_nodes]
            vw1[:, -11:] = vf_weights1[:, -11:]
            vb1 = vf_bias1
            
            pw2 = pi_weights2
            pb2 = pi_bias2
            vw2 = vf_weights2
            vb2 = vf_bias2
            
            pw3[:len(self.survived_act_nodes), :] = pi_weights3[self.survived_act_nodes, :]
            pb3[:len(self.survived_act_nodes)] = pi_bias3[self.survived_act_nodes]
            vw3 = vf_weights3
            vb3 = vf_bias3
            
            policy_weights_new = []
            policy_weights_new += pw1.reshape(-1, ).tolist()
            policy_weights_new += pb1.tolist()
            policy_weights_new += pw2.tolist()
            policy_weights_new += pb2.tolist()
            policy_weights_new += vw1.reshape(-1, ).tolist()
            policy_weights_new += vb1.tolist()
            policy_weights_new += vw2.tolist()
            policy_weights_new += vb2.tolist()
            policy_weights_new += pw3.reshape(-1, ).tolist()
            policy_weights_new += pb3.tolist()
            policy_weights_new += vw3.reshape(-1, ).tolist()
            policy_weights_new += [vb3]
        else:
            # insert the entire old weights
            pass

        return policy_weights_new

    def transform_parameter_file(self):
        parameter_old = self.parameter_old
        structure_edges_old = parameter_old["structure_edges"]
        mu_old = parameter_old["structure_weights"]["mu"]
        sigma_old = parameter_old["structure_weights"]["sigma"]
        self.policy_old = np.array(parameter_old['policy_weights'])

        parameter_new_format = self.parameter_new_format
        mu_format = parameter_new_format["structure_weights"]["mu"]
        sigma_format = parameter_new_format["structure_weights"]["sigma"]

        max_num_limbs_old = self.max_num_limbs_old
        self.structure_tree = [[] for i in range(max_num_limbs_old + 1)]
        self.dofs = np.zeros(max_num_limbs_old)
        for parent, child, dof in structure_edges_old:
            self.structure_tree[parent].append(child)
            self.dofs[child] = dof

        self.mu_old_list = np.array(mu_old).reshape([-1, 9]).tolist()
        self.sigma_old_list = np.array(sigma_old).reshape([-1, 9]).tolist()
        self.mu_format_list = np.array(mu_format).reshape([-1, 9]).tolist()
        self.sigma_format_list = np.array(sigma_format).reshape([-1, 9]).tolist()

        self.structure_edges_new = []
        self.transform_structure_para(0, -1, -1)
        structure_edges_new = self.structure_edges_new

        mu_new = sum(self.mu_format_list, [])
        sigma_new = sum(self.sigma_format_list, [])
        policy_weights_new = self.make_policy_para()

        parameter_new = {"structure_edges" : structure_edges_new, "structure_weights" : {"mu" : mu_new, "sigma" : sigma_new}, "policy_weights" : policy_weights_new}

        # with open(os.path.join("zoo", "walker", f"{os.path.basename(self.log_old)}_parameter{self.parameter_filename}_max{self.max_num_limbs_new}.json"), 'w') as f:
        #     json.dump(parameter_new, f)

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "parameter_transformed.json"), 'w') as f:
            json.dump(parameter_new, f)


if __name__ == '__main__':
    # log_old =
    # cfg_name =
    log_old = os.path.join("log", "curriculum_first_half_nofail_1")#"log/curriculum_free_1_first_half"
    parameter_filename = "best"  # have to change
    cfg_name = "ewalker_iso6.json"
    output_dir = os.path.join("log", "transform_test")#"log/curriculum_free_1_second_half"
    # Transform(log_old, parameter_filename, cfg_name, output_dir).transform_parameter_file()
    t = Transform(log_old, parameter_filename, cfg_name, output_dir)
    t.transform_parameter_file()
    
    san = np.array(t.survived_act_nodes)
    son = np.array(t.survived_obs_nodes)
    
    np.savetxt(os.path.join(output_dir, 'survived_act_nodes.csv'), san, delimiter=',')
    np.savetxt(os.path.join(output_dir, 'survived_obs_nodes.csv'), son, delimiter=',')
    
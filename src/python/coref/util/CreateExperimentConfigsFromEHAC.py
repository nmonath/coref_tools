"""
Copyright (C) 2018 University of Massachusetts Amherst.
This file is part of "coref_tools"
http://github.com/nmonath/coref_tools
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from coref.util.Config import Config
import sys
import os

if __name__ == "__main__":
    """
    From a base EHAC config file, create config files for:
    
        Incremental Greedy - (Exact) - Add to Mention
        Incremental Greedy - (Exact) - Add Anywhere
        Incremental Greedy - (NSW) - Add to Mention
        Incremental Greedy - (NSW) - Add Anywhere
        
        PERCH - (Exact) - Add to Mention
        PERCH - (Exact) - Add Anywhere
        PERCH - (NSW) - Add to Mention
        PERCH - (NSW) - Add Anywhere
        
        GERCH - (Exact) - Add to Mention
        GERCH - (Exact) - Add Anywhere
        GERCH - (NSW) - Add to Mention
        GERCH - (NSW) - Add Anywhere

        BGERCH - (Exact) - Add to Mention
        BGERCH - (Exact) - Add Anywhere
        BGERCH - (NSW) - Add to Mention
        BGERCH - (NSW) - Add Anywhere
    
        MBEHAC - (Exact) - MBSizes of 10, 25, 50, 100 
    
    """
    configfile = sys.argv[1]
    nnk = int(sys.argv[2])
    nsw_r = int(sys.argv[3])

    algs = ['acoref-greedy', 'acoref-perch', 'acoref-grinch']
    exact_nn = [False]
    atms = [True]
    exact_best_pws = [True,False]

    exp_dir = os.path.dirname(configfile)

    def filename(alg,exact,atm,nn_k,nsw_r):
        return "config_%s_exact_%s_nnk_%s_nswr_%s_ebp_%s.json" % (alg,exact,nn_k,nsw_r,exact_best_pw)

    for alg in algs:
        for exact in exact_nn:
            for atm in atms:
                for exact_best_pw in exact_best_pws:
                    base = Config(configfile)
                    base.config_name = os.path.join(exp_dir,filename(alg,exact,atm,nnk,nsw_r))
                    base.clustering_scheme = alg
                    base.exact_nn = exact
                    base.add_to_mention = atm
                    base.nn_k = nnk
                    base.nsw_r = nsw_r
                    base.write_every_tree = False
                    base.fast_graft = False
                    base.beam = 5
                    base.exact_best_pairwise = exact_best_pw
                    base.save_config(exp_dir,filename(alg,exact,atm,nnk,nsw_r))

    # mbsizes = [10, 25, 50,100]
    # for mbsize in mbsizes:
    #     fname = "config_mbehac_mbsize_%s.json" % mbsize
    #     base = Config(configfile)
    #     base.config_name = os.path.join(exp_dir, fname)
    #     base.clustering_scheme = 'mbehac'
    #     base.mbsize = mbsize
    #     base.write_every_tree = False
    #     base.save_config(exp_dir, fname)
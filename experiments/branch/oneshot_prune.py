import copy
from foundations.paths import hparams
from experiments.branch import base
import models.registry
import pruning.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train

class Branch(base.Branch):
    def branch_function(self, ):
        # Establish a simplified version of the main experiment runners

        self.main_desc.save(self.zero_branch_root)
        if self.main_desc.pretrain_training_hparams: self._pretrain()

        self._establish_initial_weights(0)
        # train from scratch experiments need to reinitialize model every time brfore training
        if self.main_experiment == 'scratch': self._establish_initial_weights(self.level)


        self._train_before_prune()

        # Alter the oneshot pruning fraction to match the iterative pruning fraction
        base_pruning_fraction = self.main_desc.pruning_hparams.pruning_fraction
        oneshot_pruning_fraction = 1 - (1 - base_pruning_fraction)**self.level
        pruning_hparams = copy.deepcopy(self.main_desc.pruning_hparams)
        dataset_hparams = copy.deepcopy(self.main_desc.dataset_hparams)
        pruning_hparams.pruning_fraction = oneshot_pruning_fraction
        self._prune(pruning_hparams, dataset_hparams)

        # Retrain the pruned model
        self._retrain()

    def _pretrain(self):
        location = self.main_desc.run_path(self.replicate, 'pretrain')
        if models.registry.exists(location, self.main_desc.pretrain_end_step): return

        if self.verbose: print('-'*82 + '\nPretraining\n' + '-'*82)
        model = models.registry.get(self.main_desc.model_hparams, outputs=self.main_desc.pretrain_outputs, 
                                    pruning_strategy = self.main_desc.pruning_hparams.pruning_strategy)
        train.standard_train(model, location, self.main_desc.pretrain_dataset_hparams, self.main_desc.pretrain_training_hparams,
                             verbose=self.verbose)

    def _establish_initial_weights(self, _level):
        location = self.main_desc.run_path(self.replicate, _level, self.experiment_name)
        if models.registry.exists(location, self.main_desc.train_start_step): return
        
        # If initial weighs exist in main folder, adopt it for oneshot pruning branch
        main_init_loc = self.main_desc.run_path(self.replicate, _level)
        if models.registry.exists(main_init_loc, self.main_desc.train_start_step):
            new_model = models.registry.load(main_init_loc, self.main_desc.train_start_step, 
                                             self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                             self.main_desc.pruning_hparams.pruning_strategy)
        else:
            new_model = models.registry.get(self.main_desc.model_hparams, outputs=self.main_desc.train_outputs, 
                                            pruning_strategy=self.main_desc.pruning_hparams.pruning_strategy)

            # If there was a pretrained model, retrieve its final weights and adapt them for training (only for level 0).
            if self.main_desc.pretrain_training_hparams is not None and _level == 0:
                pretrain_loc = self.main_desc.run_path(self.replicate, 'pretrain')
                old = models.registry.load(pretrain_loc, self.main_desc.pretrain_end_step,
                                           self.main_desc.model_hparams, self.main_desc.pretrain_outputs, 
                                           self.main_desc.pruning_hparams.pruning_strategy)
                state_dict = {k: v for k, v in old.state_dict().items()}

                # Select a new output layer if number of classes differs.
                if self.main_desc.train_outputs != self.main_desc.pretrain_outputs:
                    state_dict.update({k: new_model.state_dict()[k] for k in new_model.output_layer_names})

                new_model.load_state_dict(state_dict)
            # Save the init model in main folder
            new_model.save(main_init_loc, self.main_desc.train_start_step)
        
        # Save the init model in oneshot branch folder
        new_model.save(location, self.main_desc.train_start_step)

    def _train_before_prune(self):

        # If a trained model doesn't exist in level_0 branch folder nor the level_0 main folder, train it from initialization
        if models.registry.exists(self.zero_branch_root, self.main_desc.train_end_step): return
        else: 
            zero_main_root = self.main_desc.run_path(self.replicate, 0)
            if not models.registry.exists(zero_main_root, self.main_desc.train_end_step):
                if self.verbose:
                    print('-'*82 + '\nPruning Level 0\n'+ '-'*82)
                # train the model from initialization
                init_model = models.registry.load(self.zero_branch_root, self.main_desc.train_start_step,
                                                  self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                                  self.main_desc.pruning_hparams.pruning_strategy)
                train.standard_train(init_model, self.zero_branch_root, self.main_desc.dataset_hparams, self.main_desc.training_hparams,
                                        start_step=self.main_desc.train_start_step, verbose=self.verbose)
            else:
                trained_model = models.registry.load(zero_main_root, self.main_desc.train_end_step,
                                                     self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                                     self.main_desc.pruning_hparams.pruning_strategy)
                trained_model.save(self.zero_branch_root, self.main_desc.train_end_step)
    

    def _retrain(self, ):
        # retrain the pruned model 

        location = self.branch_root
        if models.registry.exists(location, self.main_desc.train_end_step): return

        if self.verbose:
            print('-'*82 + '\nPruning Level {}\n'.format(self.level) + '-'*82)

        if self.level != 0:
            if self.main_experiment == 'finetune':
                # If main experiment is 'finetune', use the trained weights from level_0 at the train_end_step 
                model = models.registry.load(self.zero_branch_root, self.main_desc.train_end_step,
                                             self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                             self.main_desc.pruning_hparams.pruning_strategy)
                training_hp = self.main_desc.finetuning_hparams
                start_step = self.main_desc.finetune_start_step

            elif self.main_experiment == 'lottery':
                # If main experiment is 'lottery', use the original initialization from level_0 at the train_start_step
                model = models.registry.load(self.zero_branch_root, self.main_desc.train_start_step,
                                             self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                             self.main_desc.pruning_hparams.pruning_strategy)
                training_hp = self.main_desc.training_hparams
                start_step = self.main_desc.train_start_step

            elif self.main_experiment == 'rewindLR':
                # If main experiment is 'rewindLR', use the trained weights from level_0 at the train_end_step 
                model = models.registry.load(self.zero_branch_root, self.main_desc.train_end_step,
                                             self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                             self.main_desc.pruning_hparams.pruning_strategy)
                training_hp = self.main_desc.training_hparams
                start_step = self.main_desc.train_start_step

            elif self.main_experiment == 'scratch':
                # If main experiment is 'scratch', use the randomly reinitialized weights from [self.level] at the train_sart_step 
                model = models.registry.load(location, self.main_desc.train_start_step,
                                             self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                             self.main_desc.pruning_hparams.pruning_strategy)
                training_hp = self.main_desc.training_hparams
                start_step = self.main_desc.train_start_step

            freeze_pruned_weights = self.main_desc.pruning_hparams.freeze_pruned_weights
            if freeze_pruned_weights == 'init':
                model_for_reset = models.registry.load(self.zero_branch_root, self.main_desc.train_end_step,
                                                    self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                                    self.main_desc.pruning_hparams.pruning_strategy)
            elif freeze_pruned_weights == 'final'  or freeze_pruned_weights == 'permuted' :
                model_for_reset = models.registry.load(self.zero_branch_root, self.main_desc.train_end_step,
                                                    self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                                    self.main_desc.pruning_hparams.pruning_strategy)   
            else:
                model_for_reset = None
            pruned_model = PrunedModel(model, Mask.load(location), model_for_reset, freeze_pruned_weights)
            pruned_model.save(location, self.main_desc.train_start_step)

            train.standard_train(pruned_model, location, self.main_desc.dataset_hparams, training_hp,
                                start_step=start_step, verbose=self.verbose)

    def _prune(self, pruning_hparams: hparams, dataset_hparams:hparams=None):
        new_location = self.branch_root
        old_location = self.zero_branch_root
        if Mask.exists(new_location): return

        # if not Mask.exists(old_location):
        #     Mask.ones_like(models.registry.get(self.main_desc.model_hparams, outputs=self.main_desc.train_outputs)).save(old_location)
        
        if self.level != 0:
            ones_mask = Mask.ones_like(models.registry.get(self.main_desc.model_hparams, outputs=self.main_desc.train_outputs,
                                                           pruning_strategy = self.main_desc.pruning_hparams.pruning_strategy))
            model = models.registry.load(old_location, self.main_desc.train_end_step,
                                         self.main_desc.model_hparams, self.main_desc.train_outputs, 
                                         self.main_desc.pruning_hparams.pruning_strategy)
            pruning.registry.get(pruning_hparams)(model, ones_mask, dataset_hparams).save(new_location)


    @staticmethod
    def description():
        return "Mask the same level 0 model to a set of sparsity levels with one-shot pruning. Such results could also be obtained by using the main experiment runners "\
               "with random seeds fixed to constant, levels set to 1 and pruning_fraction adjusted accordingly. Nevertheless, this approach could save you from training "\
                "the initial model for multiple times "

    @staticmethod
    def name():
        return 'oneshot_prune'
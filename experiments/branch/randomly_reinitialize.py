from experiments.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train


class Branch(base.Branch):
    def branch_function(self, start_at_step_zero: bool = False):
        # model_for_reset = models.registry.load(self.zero_branch_root, self.main_desc.train_end_step,
        #                                         self.main_desc.model_hparams, self.main_desc.train_outputs)
        # freeze_pruned_weights = self.main_desc.pruning_hparams.freeze_pruned_weights
        model = PrunedModel(models.registry.get(self.main_desc.model_hparams, outputs=self.main_desc.train_outputs,
                            pruning_strategy = self.main_desc.pruning_hparams.pruning_strategy), Mask.load(self.level_root))
        start_step = self.main_desc.str_to_step('0it') if start_at_step_zero else self.main_desc.train_start_step
        Mask.load(self.level_root).save(self.branch_root)
        train.standard_train(model, self.branch_root, self.main_desc.dataset_hparams,
                             self.main_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Randomly reinitialize the model."

    @staticmethod
    def name():
        return 'randomly_reinitialize'
import os.path
import timm
import numpy as np
import torch
import torchvision
from pt_constants import PTConstants
from simple_network import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager


class ChestXrayTrainer(Executor):
    def __init__(
        self,
        data_path="/dataset",
        log_path="/home/hamsteryang0/log",
        model_name="resnet18",
        lr=0.01,
        epochs=5,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None,
        pretrained_model_path=None,
    ):
        """Cifar10 Trainer handles train and submit_model tasks. During train_task, it trains a
        simple network on CIFAR10 dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
        """
        super().__init__()

        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars
        self.summary_writer = SummaryWriter(log_dir=log_path)
        self._log_dir = log_path
        self.num_rounds = 0
        self.pretrained_model_path = pretrained_model_path
        # Training setup
        #self.model = torchvision.models.resnet18()
        #self.model = resnet18(pretrained=True)
        
        # Step 2: Load the pretrained weights if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            state_dict = torch.load(self.pretrained_model_path)
            self.model = timm.create_model(model_name, pretrained=False, num_classes=2)
            if "fc.weight" in state_dict:
                del state_dict["fc.weight"]
            if "fc.bias" in state_dict:
                del state_dict["fc.bias"]
            self.model.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained model from:", pretrained_model_path)
        else:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=2)
        
        print(self.model)
        print(os.path.dirname(os.path.abspath(__file__)))
        print("======================================")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # Create Cifar10 dataset for training.
        transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_data_path = os.path.join(data_path, 'train') # 設置 chest_xray_ima
        self._train_dataset = ImageFolder(root=train_data_path, transform=transform)
        self._train_loader = DataLoader(self._train_dataset, batch_size=32, shuffle=True,num_workers=4,pin_memory=True)
        self._n_iterations = len(self._train_loader)        
        # Add this code in your __init__ method:
        val_data_path = os.path.join(data_path, 'val') # Set the path to your validation data
        self._val_dataset = ImageFolder(root=val_data_path, transform=transform)
        self._val_loader = DataLoader(self._val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {
            "train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf
        )

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(
                        fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(
                        fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(
                    v) for k, v in dxo.data.items() }
                self._local_train(fl_ctx, torch_weights, abort_signal)
                numround = self.num_rounds
                numround = numround + 1
                self.num_rounds = numround
                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self._save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy()
                               for k, v in new_weights.items() }
                #model_dict.update(new_weights)
                #self.model.load_state_dict(new_weights)
                outgoing_dxo = DXO(
                    data_kind=DataKind.WEIGHTS,
                    data=new_weights,
                    meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations},
                )
                return outgoing_dxo.to_shareable()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self._load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        finally:
            self.summary_writer.close()

    def _local_train(self, fl_ctx, weights, abort_signal):
        # Set the model weights
        #self.model.load_state_dict(state_dict=weights)
        # change  numpy.ndarray to torch.Tensor
        weights = {k: torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v.to(self.device) for k, v in weights.items()}
        # creat new state_dict and del fc.weight andfc.bias
        new_state_dict = {k: v for k, v in weights.items() if k not in ["fc.weight", "fc.bias"]}
        self.model.load_state_dict(new_state_dict, strict=False)

        summary_path = os.path.join(self._log_dir, f"{fl_ctx.get_identity_name()}_round_{self.num_rounds}")
        self.summary_writer = SummaryWriter(summary_path)
        # Basic training
        self.model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            correct_predictions = 0  # Variable to store correct predictions in each epoch
            total_samples = 0
            for i, batch in enumerate(self._train_loader):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                self.optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                
                # Calculate accuracy for the current batch
                _, predicted = torch.max(predictions.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                if i % 3000 == 0:
                    self.log_info(
                        fl_ctx, f"Epoch: {epoch}/{self._epochs}, Loss: {running_loss}"
                    )
                    self.summary_writer.add_scalar('training loss', running_loss,epoch)
                    running_loss = 0.0
            
            #self.log_info(fl_ctx, f"Epoch: {epoch}/{self._epochs}, Loss: {running_loss}")
            #self.summary_writer.add_scalar('training loss', running_loss,epoch)
            # Calculate accuracy for the current epoch
            accuracy = correct_predictions / total_samples
            self.log_info(fl_ctx, f"Epoch: {epoch}/{self._epochs}, Accuracy: {accuracy}")
            self.summary_writer.add_scalar('training accuracy', accuracy, epoch)
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                correct_predictions_val = 0
                total_samples_val = 0
                running_loss_val = 0.0   # Initialize validation running loss
                for i, batch in enumerate(self._val_loader):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                    predictions = self.model(images)
                    cost = self.loss(predictions, labels)
                    running_loss_val += cost.cpu().detach().numpy() / images.size()[0]  # Calculate validation running loss
                    _, predicted = torch.max(predictions.data, 1)
                    total_samples_val += labels.size(0)
                    correct_predictions_val += (predicted == labels).sum().item()
                val_accuracy = correct_predictions_val / total_samples_val
                val_loss = running_loss_val / len(self._val_loader)  # Calculate average validation loss
                self.log_info(fl_ctx, f"Epoch: {epoch}/{self._epochs}, Validation Accuracy: {val_accuracy},Validation loss: {val_loss}")
                self.summary_writer.add_scalar('validation accuracy', val_accuracy, epoch)
                self.summary_writer.add_scalar('validation loss', val_loss, epoch)  # Log validation loss
        # Add the final accuracy after all epochs
        final_accuracy = correct_predictions / total_samples
        self.log_info(fl_ctx, f"Final Accuracy: {final_accuracy}")
        self.summary_writer.add_scalar('final accuracy', final_accuracy, self.num_rounds)
        self.summary_writer.close()

    def _save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
            fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def _load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
            fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(
            exclude_vars=self._exclude_vars)
        return ml

import torch
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader


TASKS = [
    "mnli",
    "qqp",
    "qnli",
    "sst2",
    "stsb",
    "mrpc",
    "rte"
    ]

class ModelEvaluator:
   """
   A class to evaluate the performance of GLUE models.
   """
   def __init__(self, device):
      self.device = device
      self.datasets = {}
      print("Loading datasets...")
      for task in TASKS:
         if task == "mnli":
            split = "validation_matched"
         else:
            split = "validation"
         dataset = load_dataset("glue", task, split=split)
         self.datasets[task] = dataset
   
   def evaluate(self, model, tokenizer, task, batch_size=8, change_id_mnli=False):
      """
      Evaluates the Given Model and Returns the Metrics for the specified GLUE/SQUAD task

      Args:
      model: The model that needs to be evaluated
      tokenizer
      task: The GLUE task that the model needs to be evaluated on

      Returns:
      A dictionary containing the metrics score
      """

      assert task in TASKS, f"Task name must be one of {TASKS}"
      assert task in self.datasets.keys(), f"Dataset for task {task} not found"
      assert batch_size == None or batch_size > 0, "Iterations must be a positive integer"

      testing_set = self.datasets[task]

      self.model = model
      model.eval()
      
      self.tokenizer = tokenizer
      self.task = task
      self.batch_size = batch_size
      self.change_id_mnli = change_id_mnli

      test_dataloader = DataLoader(testing_set, batch_size=batch_size, shuffle=False)

      predicted_values = []
      referenced_values = []
      
      print("Getting Predictions...")
      if task == 'sst2':
         param1, _ , _ = testing_set.features.keys()
      else:
         param1, param2, _, _ = testing_set.features.keys()

      for batch in tqdm(test_dataloader):
         if task == 'sst2':
            referenced_value, predicted_value = self.get_batch_prediction_glue(batch, param1)
         else:
            referenced_value, predicted_value = self.get_batch_prediction_glue(batch, param1, param2)

         referenced_values.extend(referenced_value)
         predicted_values.extend(predicted_value)

      glue_metric = load('glue', task)
      results = glue_metric.compute(predictions=predicted_values, references=referenced_values)

      return results

   def get_batch_prediction_glue(self, batch, param1, param2=None):

         if param2:
            glue_input = self.tokenizer(batch[param1], batch[param2], return_tensors='pt', 
               padding=True, truncation=True)
         else:
            glue_input = self.tokenizer(batch[param1], return_tensors='pt', 
               padding=True, truncation=True)
         glue_input = glue_input.to(self.model.device)
        #  print("batch label:", batch['label'])
         with torch.no_grad():
            output = self.model(**glue_input).logits
            
         if self.task == "stsb":
          predicted_id = output.squeeze().tolist()
         else:
          predicted_id = torch.argmax(output,dim=1).tolist()
         
         if self.change_id_mnli:
           # textattack/bert-base-uncased-MNLI uses different labels compared to the dataset
           # 2 instead of 0
           # 0 instead of 1
           # 1 instead of 2
           predicted_id = [1 if elem==2 else 0 if elem==1 else 2 for elem in predicted_id ]
         
        #  print("batch prediction", predicted_id)
        #  print("correct prediction", batch['label'].tolist())
         return batch['label'].tolist(), predicted_id

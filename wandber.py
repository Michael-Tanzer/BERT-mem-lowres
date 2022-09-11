import wandb
import shutil


class Wandber:

    def __init__(self, on):
        self.on = on

    def init(self, run_name, parameters):
        if self.on:
            wandb.init(project="<wandb-project-name>",
                       name=run_name,
                       config=parameters)

    def watch(self, model):
        if self.on:
            wandb.watch(model)

    def log_training_step(self, parameters_dict, step):
        if self.on:
            parameters_dict = {f"training-{k}": v for k, v in parameters_dict.items()}
            wandb.log(parameters_dict, step=step)

    def log_validation_step(self, parameters_dict, step):
        if self.on:
            parameters_dict = {f"validation-{k}": v for k, v in parameters_dict.items()}
            wandb.log(parameters_dict, step=step)

    def log_summary_metrics(self, parameters_dict, name):
        if self.on:
            name = "validation" if name == "dev" else name
            for k, v in parameters_dict.items():
                wandb.run.summary[f"{name}-{k}"] = v

    def add_code(self, code_directory="<code_folder>",
                 zip_directory="<some_temp_folder>", filename="source_code"):
        if self.on:
            zip_out = shutil.make_archive(zip_directory + filename, 'zip', root_dir=code_directory)
            wandb.save(zip_out)

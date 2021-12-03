from azureml.core import Workspace,Dataset,Environment,Experiment
from azureml.core.model import Model,InferenceConfig
from azureml.core.webservice import AciWebservice,Webservice
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication
import os

#ws = Workspace.from_config()
svc_pr = ServicePrincipalAuthentication(
    tenant_id="c72bdf19-b6fa-4116-bfed-ea148f518e88",
    service_principal_id="22d732ca-f5da-47d5-82e8-909ca6a3f0a5",
    service_principal_password="dGI7Q~UImpGTNK2kyfQ59BuejQvqsYN3sNz9N") 
ws = Workspace.get(name="zf-bd-aml-hp",
                   subscription_id="3ec8454a-1092-4081-ad37-74ea5816abaf", 
                   resource_group="ZF-BD-HP",
                   auth=svc_pr)
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

experiment_name = 'train-a-devops'
exp = Experiment(workspace=ws, name=experiment_name)

pytorch_env = Environment.from_conda_specification(name = 'pytorch-env', file_path = './conda_dependencies.yml')
model=Model(ws,"pytorch-BD")

script_folder = os.getcwd()
print(script_folder)
inference_config = InferenceConfig(source_directory=script_folder,entry_script="pred_score.py",environment=pytorch_env)

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={'framework':'pytorch'},
                                               description='ZF POC with PyTorch')

service = Model.deploy(workspace=ws, 
                           name='zf-bbdd-aci', 
                           models=[model], 
                           inference_config=inference_config, 
                           deployment_config=aciconfig,
                           overwrite=True)
service.wait_for_deployment(True)
#print(service.get_logs())
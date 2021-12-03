from azureml.core import Workspace,Dataset,Experiment,ScriptRunConfig,Environment
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, help='dataset name')
parser.add_argument('-f')
args = parser.parse_args()
print(args)

print("Dataset name: " + args.dataset_name)

#ws = Workspace.from_config()

##svc_pr_password = os.environ.get("AZUREML_PASSWORD")
# use generated Service Principal Key in Portal to login
svc_pr = ServicePrincipalAuthentication(
    tenant_id="c72bdf19-b6fa-4116-bfed-ea148f518e88",
    service_principal_id="22d732ca-f5da-47d5-82e8-909ca6a3f0a5",
    service_principal_password="dGI7Q~UImpGTNK2kyfQ59BuejQvqsYN3sNz9N") 
ws = Workspace.get(name="zf-bd-aml-hp",
                   subscription_id="3ec8454a-1092-4081-ad37-74ea5816abaf", 
                   resource_group="ZF-BD-HP",
                   auth=svc_pr)

# use interactive dynamic codes to login
#interactive_auth = InteractiveLoginAuthentication(tenant_id="c72bdf19-b6fa-4116-bfed-ea148f518e88")
#ws = Workspace.from_config(auth=interactive_auth)
print(ws.name, ws.resource_group, ws.subscription_id, sep='\n')

dataset_name = args.dataset_name #'hdfdata'
# Get a dataset by name
hdf_ds = Dataset.get_by_name(workspace=ws, name=dataset_name)
hdf_ds.to_path()

experiment_name = 'train-a-devops'
exp = Experiment(workspace=ws, name=experiment_name)

# choose a name for your cluster
compute_name = os.environ.get('AML_COMPUTE_CLUSTER_NAME', 'cpu-a-cluster')
compute_min_nodes = os.environ.get('AML_COMPUTE_CLUSTER_MIN_NODES', 0)
compute_max_nodes = os.environ.get('AML_COMPUTE_CLUSTER_MAX_NODES', 1)
# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
vm_size = os.environ.get('AML_COMPUTE_CLUSTER_SKU', 'Standard_DS11_v2')
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found compute target. just use it. ' + compute_name)
else:
    print('creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                min_nodes=compute_min_nodes, 
                                                                max_nodes=compute_max_nodes)
    # create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)   
    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)   
     # For a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())

pytorch_env = Environment.from_conda_specification(name = 'pytorch-env', file_path = './conda_dependencies.yml')

#print("DATA PATH: " + thisdataset.to_path()->list)
script_folder = os.getcwd()
print(script_folder)
src = ScriptRunConfig(source_directory=script_folder,
                      script='Trainer.py',
                      arguments=['--data_path',hdf_ds.as_mount()],
                      compute_target=compute_target,
                      environment=pytorch_env)

#import azureml._restclient.snapshots_client
#azureml._restclient.snapshots_client. SNAPSHOT_MAX_SIZE_BYTES =1000000000000
run = exp.submit(src)
run.wait_for_completion(show_output=True)

model = run.register_model(model_name='pytorch-devops-BD', model_path= 'outputs/thismodel.pt')

from azureml.core.run import Run
from azureml.core import Workspace,Dataset
run = Run.get_context()

print(run)
ws = Workspace.get(name="zf-poc-hp",
                   subscription_id="2bd536a1-23a8-4193-be1e-219890bbd881", resource_group="zf-aml-hp")
print(ws.name, ws.resource_group, ws.subscription_id, sep='\n')
dataset_name = 'hdfdata'
hdf_ds = Dataset.get_by_name(workspace=ws, name=dataset_name)
hdf_ds.to_path()
print(hdf_ds)
print(hdf_ds.to_path())
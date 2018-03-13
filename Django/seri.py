from django.core import serializers
import json



with open('discussion.json') as f:
	data_ls = json.load(f)
	data_dict = {}

	for i in range(len(data_ls)):
		data_dict[i] = data_ls[i]


#for obj in serializers.deserialize("json", data_dict):
#	obj.save()


def load_fixture(apps, schema_editor):
    from django.conf import settings
    if settings.TESTS_RUNNING:
        # Currently we don't have any test relying to borders, we skip
        # to accelerate tests
        return

    fixture_file = os.path.join(fixture_dir, fixture_filename)

	objects = serializers.deserialize('json', data_dict, ignorenonexistent=True)
	for obj in objects:
		obj.save()
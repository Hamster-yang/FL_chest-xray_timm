import timm
available_models = timm.list_models()
print(available_models)

model_name = available_models
# Convert list to a formatted string
formatted_string = "\n".join(model_name)
print(formatted_string)
# Save the formatted string to a text file

file_path = "./model_list_timm0-9-2.txt"
with open(file_path, 'w') as file:
    file.write(formatted_string)

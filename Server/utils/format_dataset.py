def format_dataset(datasetPath):
    with open(datasetPath, 'r') as f:
        ndataset = ""
        columns = "review;sentiment"
        for i, line in enumerate(f.readlines()):
            if (i == 0):
                continue
            parts = line.split(';')
            line = f"{parts[1]};{parts[2]}"
            ndataset = f"{ndataset}{line}"
        ndataset = f"{columns}\n{ndataset}"
        with open('./Server/DB/formatDataset.txt', 'w') as nf:
            nf.write(ndataset)
            

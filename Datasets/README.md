# Datasets

## 00 Random Resources
Random DBpedia resources for 100 rdf:types, scraped via SPARQL.
- 10000 per rdf:type
- 50000 per rdf:type (not all contain 50000 resources)
```python
{
  rdf_type_01:
    [resource_link_01,
    resource_link_02,
    ...],
  rdf_type_02:
    [...]
  ...
}
```
## 01 Datests - used in training
The actual datasets used for training. The .zip files in this folder do not contain images, but are link collections to links in a Google Cloud Storage bucket.
```
input:	image-link
label:	rdf:type
```

## 02 Datasets - Wikipedia links
This folder contains datasets of direkt links to images on Wikipedia. They were collected subsequently and are a equivalent representation of the datasets used in training. They contain other images and have a different image count distribution per rdf:type. The rdf:types are the same.
```
input:	image-link
label:	rdf:type
```


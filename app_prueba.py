from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from embeddings_utils_1 import generate_image_embeddings, get_image_paths_from_directory


def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise


def create_collection(name, fields, description, consistency_level="Strong"):
    schema = CollectionSchema(fields, description)
    collection = Collection(name, schema, consistency_level=consistency_level)
    return collection


def insert_data(collection, entities):
    insert_result = collection.insert(entities)
    collection.flush()
    print(
        f"Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}")
    return insert_result


def create_index(collection, field_name, index_type, metric_type, params):
    index = {"index_type": index_type,
             "metric_type": metric_type, "params": params}
    collection.create_index(field_name, index)
    print(f"Index '{index_type}' created for field '{field_name}'.")


def search_and_query(collection, search_vectors, search_field, search_params):
    collection.load()

    # Vector search
    result = collection.search(
        search_vectors, search_field, search_params, limit=3, output_fields=["source"])
    print_search_results(result, "Vector search results:")


def print_search_results(results, message):
    print(message)
    for hits in results:
        for hit in hits:
            print(f"Hit: {hit}, source field: {hit.entity.get('source')}")


def delete_entities(collection, expr):
    collection.delete(expr)
    print(f"Deleted entities where {expr}")


def drop_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Dropped collection '{collection_name}'.")


# Main
dim = 2048  # Ajusta la dimensión según la salida del modelo de imágenes

connect_to_milvus()

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR,
                is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
collection = create_collection(
    "hello_milvus_images", fields, "Collection for image embeddings demo")

# Rutas de las imágenes
general_folder = 'Dataset'
image_paths = get_image_paths_from_directory(general_folder)

embeddings = generate_image_embeddings(image_paths)
entities = [
    [str(i) for i in range(len(image_paths))],
    [str(path) for path in image_paths],
    embeddings
]
insert_result = insert_data(collection, entities)

create_index(collection, "embeddings", "IVF_FLAT", "L2", {"nlist": 128})

# Supongamos que tenemos una imagen de consulta para búsqueda
#query_image_path = 'query/query/0001_c1_000260.jpg'
#query_embedding = generate_image_embeddings([query_image_path])[0]
#search_and_query(collection, [query_embedding], "embeddings", {
 #                "metric_type": "L2", "params": {"nprobe": 10}})

#delete_entities(
 #   collection, f'pk in ["{insert_result.primary_keys[0]}" , "{insert_result.primary_keys[1]}"]')
#drop_collection("hello_milvus_images")
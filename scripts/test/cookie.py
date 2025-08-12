import requests


url = "https://api.wandb.ai/graphql"
cookie = "wandb=MTc1Mzc2OTQ3M3xEdi1oQkFFQ182SUFBUkFCRUFBQUp2LWlBQUVHYzNSeWFXNW5EQXdBQ25ObGMzTnBiMjVmYVdRRmFXNTBOalFFQlFEOW5SV298In9fJAAX3f0GcOyHotHA7TnzUVPDqURhs_60b4a1-LQ="
headers = {
    "Content-Type": "application/json",
    "Cookie": cookie,
    "Origin": "https://wandb.ai",
    "Referer": "https://wandb.ai/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}

query = """
query Runs($entity: String!, $project: String!, $first: Int) {
  project(entityName: $entity, name: $project) {
    runs(first: $first) {
      edges {
        node {
          name
          displayName
          summaryMetrics
        }
      }
    }
  }
}
"""

variables = {
    "entity": "rhys-newbury",
    "project": "diffuse_keypoints_test_fr",
    "first": 100,
}

response = requests.post(
    url, headers=headers, json={"query": query, "variables": variables}
)
print(response.json())


res = requests.post(url, headers=headers, json=query)
print(res.json())

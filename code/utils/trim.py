import json

def listify (item):
    if not isinstance(item, list):
        return [item]
    else:
        return item
    
def jsonify(obj):
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            return obj
    return obj

def trim_json(obj):
    obj = jsonify(obj)
    objType = obj["@type"] if "@type" in obj else ["Thing"]
    if not isinstance(objType, list):
        objType = [objType]
    if (objType == ["Thing"]):
        return obj
    if ("Recipe" in objType):
        return trim_recipe(obj)
    if ("Movie" in objType or "TVSeries" in objType):
        return trim_movie(obj)
    return obj

def trim_json_hard(obj):
    obj = jsonify(obj)
    objType = obj["@type"] if "@type" in obj else ["Thing"]
    if not isinstance(objType, list):
        objType = [objType]
    if (objType == ["Thing"]):
        return obj
    if ("Recipe" in objType):
        return trim_recipe_hard(obj)
    if ("Movie" in objType or "TVSeries" in objType):
        return trim_movie(obj, hard=True)
    return obj
   

def trim_recipe(obj):
    obj = jsonify(obj)
    items = collateObjAttr(obj)
    js = {}
    skipAttrs = ["mainEntityOfPage", "publisher", "image", "datePublished", "dateModified", 
                 "author"]
    for attr in items.keys():
        if (attr in skipAttrs):
            continue
        js[attr] = items[attr]
    return js

def trim_recipe_hard(obj):
    items = collateObjAttr(obj)
    js = {}
    skipAttrs = ["mainEntityOfPage", "publisher", "image", "datePublished", "dateModified", "review",
                 "author", "recipeYield", "recipeInstructions", "nutrition"]
    for attr in items.keys():
        if (attr in skipAttrs):
            continue
        js[attr] = items[attr]
    return js



def trim_movie(obj, hard=False):
    items = collateObjAttr(obj)
    js = {}
    skipAttrs = ["mainEntityOfPage", "publisher", "image", "datePublished", "dateModified", "author", "trailer"]
    if (hard):
        skipAttrs.extend(["actor", "director", "creator", "review"])
    for attr, values in items.items():
        if attr in skipAttrs:
            continue
        elif attr in ("actor", "director", "creator"):
            for person in values:
                if isinstance(person, dict) and "name" in person:
                    js.setdefault(attr, []).append(person["name"])
        elif attr == "review":
            for review in values:
                if isinstance(review, dict) and "reviewBody" in review:
                    js.setdefault(attr, []).append(review["reviewBody"])
        else:
            js[attr] = values
    return js

def collateObjAttr(obj):
    items = {}
    for attr, value in obj.items():
        if attr not in items:
            items[attr] = []
        if isinstance(value, list):
            items[attr].extend(value)
        else:
            items[attr].append(value)
    return items

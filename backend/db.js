import { JSONFilePreset } from "lowdb/node";

// Read or create objects_db.json
const defaultData = { objects: [] };
const db = await JSONFilePreset("objects_db.json", defaultData);

const db_test = await JSONFilePreset("objects_db_test.json", defaultData);

/**
// Update db.json
// await db.update(({ objects }) => objects.push("hello world again"));

// Alternatively you can call db.write() explicitely later
// to write to db.json
db.data.objects.push("hello world 3");
await db.write();
 */

export const addObjectToDb = async (object, test = true) => {
  if (test === true) {
    await db_test.update(({ objects }) => objects.push(object));
  } else {
    await db.update(({ objects }) => objects.push(object));
  }
};

export const checkForId = (id, test = true) => {
  if (test === true) {
    const { objects } = db_test.data;
    if (objects.find((object) => object.objectId === id)) {
      return true;
    }
  } else {
    const { objects } = db.data;
    if (objects.find((object) => object.objectId === id)) {
      return true;
    }
  }
  return false;
};

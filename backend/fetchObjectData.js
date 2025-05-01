import axios from "axios";
import fs from "fs";
import path from "path";

import {
  createObjectUrl,
  testOutputFolder,
  testObjectId,
  testObjectIds,
  testUrlFile,
} from "./utils.js";

import { addObjectToDb, checkForId } from "./db.js";

/**
 * function to fetch object data and save to json
 */
export const fetchObjectData = async ({ objectId }) => {
  if (!objectId) {
    console.log("missing object id");
    return;
  }

  // return if objectid already saved
  if (checkForId(objectId) === true) return;

  try {
    const url = createObjectUrl(objectId);
    const { data } = await axios.get(url);

    // filter out any pair of vases, etc.
    if (data["objectName"].toLowerCase() !== "vase") return;

    const imageUrl = data.primaryImageSmall || data.primaryImage;

    // filter out any that don't have an image
    if (!imageUrl) {
      console.log(`No image found for object ${objectId}`);
      return;
    }

    // save data to json
    const object = { objectId, imageUrl, data };
    await addObjectToDb(object);
  } catch (err) {
    console.log("!!!error!!! ");
    console.error(err);
  }
};

/**
 * test calls
 */
// fetchObjectData({ objectId: testObjectId });

// for (const objectId of testObjectIds) {
//   fetchObjectData({ objectId  });
// }

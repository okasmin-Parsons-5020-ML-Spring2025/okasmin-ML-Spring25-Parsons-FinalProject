import fs from "fs";
import { objectIdsFile, objectIdsFile2 } from "./utils.js";
import pLimit from "p-limit";
import { fetchObjectData } from "./fetchObjectData.js";

const concurrencyLimit = 5;
const limit = pLimit(concurrencyLimit);

/**
 * read object ids list
 */
const jsonData = fs.readFileSync(objectIdsFile2);
const ids = JSON.parse(jsonData);

// console.log(ids.slice(0, 10));

/**
 * function to loop through array of objectIds and fetch their data
 * batch fetching using p-limit
 */
const runBatch = async (objectIds) => {
  const tasks = objectIds.map((id) => limit(() => fetchObjectData({ objectId: id })));
  await Promise.all(tasks);
  console.log("✅ All ids processed");
};

/**
 * objectIdsFile2 contains 26865
 * aiming to get about 1000-2000 total so going to slice this
 */

const ids2 = ids.slice(1000, 3000);

runBatch(ids2);

import fs from "fs";
import { objectIdsFile } from "./utils.js";
import pLimit from "p-limit";
import { fetchObjectData } from "./fetchObjectData.js";

const concurrencyLimit = 5;
const limit = pLimit(concurrencyLimit);

/**
 * read object ids list
 */
const jsonData = fs.readFileSync(objectIdsFile);
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

// runBatch(ids.slice(400, 500));

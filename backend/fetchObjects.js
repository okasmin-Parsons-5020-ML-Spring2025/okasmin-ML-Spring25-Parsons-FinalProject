import { fetchObjectData } from "./fetchObjectData.js";
import { testObjectIds, testOutputFolder, testUrlFile } from "./utils.js";
// import pLimit from "p-limit";

// const concurrencyLimit = 10;
// const limit = pLimit(concurrencyLimit);

/**
 * function to loop through array of objectIds and fetch their data
 * batch fetching using p-limit
 */

for (const objectId of testObjectIds) {
  fetchObjectData({ objectId, imageFolder: testOutputFolder, filePath: testUrlFile });
}

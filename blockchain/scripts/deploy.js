import hre from "hardhat";

async function main() {
  const VoiceLedger = await hre.ethers.getContractFactory("VoiceLedger");
  const ledger = await VoiceLedger.deploy();

  await ledger.waitForDeployment();

  console.log("VoiceLedger deployed to:", await ledger.getAddress());
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

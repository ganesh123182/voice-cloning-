// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract VoiceLedger {
    struct EnrollmentRecord {
        string evidenceHash;
        uint256 timestamp;
    }

    // Mapping from enrollment_id to the record
    mapping(string => EnrollmentRecord) private enrollments;

    event EnrollmentAnchored(string indexed enrollmentId, string evidenceHash, uint256 timestamp);

    function storeHash(string memory enrollmentId, string memory evidenceHash) public {
        require(bytes(enrollments[enrollmentId].evidenceHash).length == 0, "Enrollment ID already exists");
        
        enrollments[enrollmentId] = EnrollmentRecord({
            evidenceHash: evidenceHash,
            timestamp: block.timestamp
        });

        emit EnrollmentAnchored(enrollmentId, evidenceHash, block.timestamp);
    }

    function getHash(string memory enrollmentId) public view returns (string memory) {
        return enrollments[enrollmentId].evidenceHash;
    }
}

import unittest
from subnet.remote_backend import training_submission_bytes
from subnet.artifact_budget import LEGACY


class TrainingDownloadCapacity(unittest.TestCase):
    def test_all_selected_receipts_count_without_unselected_uploads(self):
        receipts={str(i):dict(size=100+i,sha256=str(i)) for i in range(4)}
        reports={str(i):dict(accepted=[{}] if i<3 else [],submission_sha256=str(i)) for i in range(4)}
        self.assertEqual(training_submission_bytes(receipts,reports,{}),303)

    def test_invalid_size_or_substituted_receipt_refused(self):
        report={'miner':dict(accepted=[{}],submission_sha256='approved')}
        for size in (None,True,0,-1,LEGACY['compressed_bytes']+1):
            with self.assertRaises(ValueError):training_submission_bytes({'miner':dict(size=size,sha256='approved')},report,{})
        with self.assertRaises(ValueError):training_submission_bytes({'miner':dict(size=1,sha256='substituted')},report,{})

    def test_empty_or_excessive_download_population_refused(self):
        with self.assertRaises(ValueError):training_submission_bytes({}, {}, {})
        receipts={str(i):dict(size=1,sha256=str(i)) for i in range(257)}
        reports={str(i):dict(accepted=[{}],submission_sha256=str(i)) for i in range(257)}
        with self.assertRaises(ValueError):training_submission_bytes(receipts,reports,{})

if __name__=='__main__':unittest.main()

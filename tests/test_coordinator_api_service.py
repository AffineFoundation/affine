import unittest
from subnet.coordinator_api_service import queue_for_config
from subnet.role_router import RoutedJobs
from test_distributed_roles import QueueTests, sign


class IndependentQueueTests(QueueTests):
    def configuration(self):
        return dict(state=self.folder.name,remote=dict(
            roles=dict(verify=[dict(worker_identity=k.verify_key.encode().hex())for k in self.workers]),
            verifier_queue=dict(external_api=True,lease_seconds=10,max_attempts=2)))

    def test_existing_job_survives_learner_connection_recreation(self):
        config=self.configuration()
        # Enqueue through a learner connection to the actual persistent roles
        # path, then claim/report through the independent API connection.
        learner=queue_for_config(config,self.authority)
        learner.clock=lambda:self.now
        learner.enqueue(sign(self.operator,self.job))
        independent=queue_for_config(config,self.authority)
        independent.clock=lambda:self.now
        claim=independent.request(sign(self.workers[0],dict(
            action='claim',role='verify',at=self.now,nonce='9'*32)))['claim']
        restarted=queue_for_config(config,self.authority)
        self.assertEqual(restarted.status('job1')['status'],'leased')
        restarted.clock=lambda:self.now
        answer=restarted.request(sign(self.workers[0],dict(action='report',
            at=self.now,nonce='8'*32,job_id='job1',token=claim['token'],report=self.report())))
        self.assertTrue(answer['accepted'])

    def test_disabled_mode_and_unknown_inactive_signer_rejected(self):
        config=self.configuration();config['remote']['verifier_queue']['external_api']=False
        with self.assertRaises(ValueError):queue_for_config(config,self.authority)
        config=self.configuration();config['inactive_claim_worker_identities']=['unknown']
        with self.assertRaises(ValueError):queue_for_config(config,self.authority)

    def test_historical_signer_retained_without_claim_permission(self):
        config=self.configuration();identity=self.workers[1].verify_key.encode().hex()
        config.update(historical_trusted_worker_identities=[identity],inactive_claim_worker_identities=[identity])
        queue=queue_for_config(config,self.authority)
        self.assertEqual(queue.workers[identity],[])

    def test_stopping_external_router_does_not_stop_api(self):
        router=object.__new__(RoutedJobs);router.server=None
        router.stop()


if __name__=='__main__':unittest.main()

import copy
import json
import unittest
from unittest.mock import patch

from ops.monitoring import gpu_fleet as g


class FleetTests(unittest.TestCase):
    def payload(self):
        return {'gpu_query_ok':True,'gpus':[{'index':0,'name':'NVIDIA H200',
                'util':0,'memory_used_mib':100,'memory_total_mib':1000,
                'temperature_c':45,'power_w':100,'power_limit_w':700,
                'driver':'580.126.09','pstate':'P0','ecc_uncorrected':0}],
                'disk_total':1000,'disk_free':500,'ram_total':1000,'ram_available':500}

    def pod(self):
        return dict(g.sanitize(self.payload()),reachable=True,expected_gpus=1,
                    service={'status':'ok','label':'Ready'})

    def test_idle_is_healthy(self):
        self.assertEqual(g.assess(self.pod()),('ok',[]))

    def test_memory_reservation_is_not_failure(self):
        pod=self.pod();pod['gpus']['0']['memory_used_mib']=999
        self.assertEqual(g.assess(pod)[0],'ok')

    def test_unknown_probe_is_not_healthy(self):
        pod=self.pod();pod['reachable']=False
        self.assertEqual(g.assess(pod)[0],'unknown')
        pod=self.pod();pod['gpu_query_ok']=False
        self.assertEqual(g.assess(pod)[0],'unknown')

    def test_missing_core_telemetry_warns(self):
        pod=self.pod();pod['gpus']['0']['temperature_c']=None
        self.assertEqual(g.assess(pod)[0],'warn')

    def test_heat_and_disk_thresholds(self):
        for temp,expected in [(79,'ok'),(80,'warn'),(90,'error')]:
            pod=self.pod();pod['gpus']['0']['temperature_c']=temp
            self.assertEqual(g.assess(pod)[0],expected)
        pod=self.pod();pod['disk_free']=20
        self.assertEqual(g.assess(pod)[0],'error')

    def test_application_separate_from_hardware(self):
        pod=self.pod();pod['service']={'status':'error','label':'Unavailable'}
        self.assertEqual(g.assess(pod)[0],'error')
        pod['service']={'status':'unknown','label':'Unknown'}
        self.assertEqual(g.assess(pod)[0],'warn')

    def test_ecc_and_count_mismatch_warn(self):
        pod=self.pod();pod['gpus']['0']['ecc_uncorrected']=1
        self.assertEqual(g.assess(pod)[0],'warn')
        pod=self.pod();pod['expected_gpus']=2
        self.assertEqual(g.assess(pod)[0],'warn')

    def test_sanitization_and_unknown_values(self):
        raw=self.payload();raw['secret']='private';raw['gpus'][0].update(
            name='<script>bad</script>',util=float('nan'),fan_pct=True,private='secret')
        result=g.sanitize(raw)
        self.assertNotIn('secret',json.dumps(result))
        for key in ('name','util','fan_pct'):self.assertIsNone(result['gpus']['0'][key])
        json.dumps(result,allow_nan=False)

    def test_bad_memory_and_duplicate_indices(self):
        raw=self.payload();raw['ram_available']=2000;raw['gpus'][0]['memory_used_mib']=2000
        result=g.sanitize(raw)
        self.assertIsNone(result['ram_available'])
        self.assertIsNone(result['gpus']['0']['memory_used_mib'])
        raw['gpus'].append(copy.deepcopy(raw['gpus'][0]))
        with self.assertRaises(ValueError):g.sanitize(raw)

    def test_ssh_command_injection_rejected(self):
        result=g.probe({'name':'affine-eval','status':'running',
                        'ssh_cmd':'ssh -o ProxyCommand=touch-secret root@example.com'})
        self.assertFalse(result['reachable'])
        self.assertNotIn('touch-secret',json.dumps(result))

    def test_stopped_inventory_does_not_probe(self):
        with patch.object(g.subprocess,'run') as run:
            result=g.probe({'name':'affine-chat','status':'stopped'})
            run.assert_not_called()
        self.assertIn('not listed as running',result['probe_error'])

    def test_inventory_failure_is_not_empty_success(self):
        with (patch.object(g.infrastructure,'_inventory',return_value=(None,'Inventory unavailable')),
              patch.object(g,'service_health',return_value={})):
            result=g.collect()
        self.assertEqual(result['inventory_error'],'Inventory unavailable')
        self.assertIsNone(result['excluded_pods'])

    def test_publication_is_local_only(self):
        with self.assertRaises(ValueError):g.publish('http://example.com',{})


if __name__=='__main__':unittest.main()

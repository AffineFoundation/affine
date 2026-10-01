import unittest
from subnet.public_candidates import propose


class PublicCandidateTests(unittest.TestCase):
    def test_rocket_mass_derived_from_visible_inputs(self):
        prompt='A rocket has an initial mass of 6 × 10⁴ kg and burns fuel at a rate of 200 kg/s. The exhaust velocity is 2450 m/s. What is the mass of the rocket at lift off? Take g = 9.8 m/s².'
        result=propose('affine_scitext',[{'role':'user','content':prompt}])
        self.assertIn(r'\boxed{50000}',result)

    def test_public_input_variation_changes_calculation(self):
        prefix='A rocket has an initial mass of 8 × 10⁴ kg and burns fuel at a rate of 300 kg/s. The exhaust velocity is '
        suffix=' m/s. How long until it lifts off? Take g = 10 m/s².'
        a=propose('affine_scitext',[{'role':'user','content':prefix+'2000'+suffix}])
        b=propose('affine_scitext',[{'role':'user','content':prefix+'2200'+suffix}])
        self.assertNotEqual(a,b)
        self.assertIn(r'\boxed{66.6667}',a)

    def test_does_not_use_extra_gold_fields(self):
        self.assertEqual(propose('affine_scitext',[{'role':'user','content':'Unknown question','answer':'42'}]),[])

    def test_bounded_campsite_derived_solution(self):
        prompt='X T X\nrow_constraints = [1]\ncol_constraints = [1, 0, 0]'
        values=propose('affine_logic',[{'role':'user','content':prompt}])
        self.assertIn("[['C', 'T', 'X']]",values)
        self.assertEqual(propose('affine_logic',[{'role':'user','content':prompt.replace('[1]','[2]')}]),[])

    def test_unsupported_and_overlong_prompt(self):
        self.assertEqual(propose('unknown',[{'role':'user','content':'42'}]),[])
        with self.assertRaisesRegex(ValueError,'budget'):propose('affine_logic',[{'role':'user','content':'x'*100001}])

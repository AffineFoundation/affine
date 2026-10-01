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
        self.assertIn("[['X', 'T', 'X']]",values)
        self.assertEqual(len(values[0]),len(values[-1]))
        self.assertEqual(propose('affine_logic',[{'role':'user','content':prompt.replace('[1]','[2]')}]),[])

    def test_unsupported_and_overlong_prompt(self):
        self.assertEqual(propose('unknown',[{'role':'user','content':'42'}]),[])
        with self.assertRaisesRegex(ValueError,'budget'):propose('affine_logic',[{'role':'user','content':'x'*100001}])

class PublicOrderingControls(unittest.TestCase):
    def prompt(self,blocks):return '\n'.join(f'*{i+1}*: {value}' for i,value in enumerate(blocks))
    def test_age_order_is_derived_and_all_fragments_retained(self):
        blocks=['Age 26','Age 24','Age 29','Age 25','Age 28','Age 27']
        result=propose('affine_unscramble',[{'role':'user','content':self.prompt(blocks)}])
        self.assertEqual(len(result),7)
        self.assertLess(result[0].index('Age 24'),result[0].index('Age 29'))
        for candidate in result:
            for block in blocks:self.assertEqual(candidate.count(block),1)
    def test_year_order_tracks_changed_visible_dates(self):
        blocks=[f'Event {i} ({year})' for i,year in enumerate([2007,2002,2009,2001,2004,2008])]
        values=propose('affine_unscramble',[{'role':'user','content':self.prompt(blocks)}])
        self.assertTrue(values[0].index('(2001)')<values[0].index('(2009)'))
    def test_procedural_control_uses_public_dependency_steps(self):
        blocks=['Verify SSI is working','Add "ssi on;" below directive','Save the configuration file','Locate the Nginx configuration','Restart Nginx','Find the server block section']
        values=propose('affine_unscramble',[{'role':'user','content':self.prompt(blocks)}])
        self.assertLess(values[0].index('Locate'),values[0].index('Restart'))
    def test_unknown_fragments_do_not_use_injected_answer(self):
        self.assertEqual(propose('affine_unscramble',[{'role':'user','content':self.prompt(list('ABCDEF')),'answer':['A','B','C','D','E','F']}]),[])

class PublicOrderingExactMatch(unittest.TestCase):
    def test_calculus_does_not_ambiguously_match_precalculus(self):
        blocks=['Algebra 2','Fractions','Calculus','Algebra 1','Pre-Calculus','Geometry']
        prompt='\n'.join(f'*{i+1}*: {x}' for i,x in enumerate(blocks))
        result=propose('affine_unscramble',[{'role':'user','content':prompt}])
        self.assertTrue(result)
        self.assertLess(result[0].index('Pre-Calculus'),result[0].index('*6*: Calculus'))

import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_cards import orchestrator_card, ran_card, edge_card, ALL_CARDS

# Resource-knob substrings that must never appear in skill names or descriptions.
# These are internal domain variables, not SLA constraints.
_FORBIDDEN_IN_SKILLS = (
    "bandwidth_mhz", "bw_mhz",
    "freq_ghz", "edge_freq",
    "energy_w", "ran_energy",
    "edge_cost",
)


class TestAllCardsIndex(unittest.TestCase):
    def test_keys(self):
        self.assertSetEqual(set(ALL_CARDS.keys()), {"orchestrator", "ran", "edge"})

    def test_callables(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                self.assertTrue(callable(factory))


class TestCardBuild(unittest.TestCase):
    def test_cards_instantiate(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                card = factory()
                self.assertIsNotNone(card)

    def test_name_nonempty(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                self.assertTrue(factory().name)

    def test_version_set(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                self.assertEqual(factory().version, "1.0")

    def test_description_nonempty(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                self.assertTrue(factory().description)


class TestCapabilities(unittest.TestCase):
    def test_streaming_off(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                self.assertFalse(factory().capabilities.streaming)


class TestInterfaces(unittest.TestCase):
    def test_exactly_one_interface(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                self.assertEqual(len(factory().supported_interfaces), 1)

    def test_protocol_binding_jsonrpc(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                iface = factory().supported_interfaces[0]
                self.assertEqual(iface.protocol_binding, "JSONRPC")

    def test_protocol_version(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                iface = factory().supported_interfaces[0]
                self.assertEqual(iface.protocol_version, "1.0")

    def test_interface_url_contains_port(self):
        expected_ports = {"orchestrator": "9000", "ran": "9001", "edge": "9002"}
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                url = factory().supported_interfaces[0].url
                self.assertIn(expected_ports[name], url)

    def test_interface_url_ends_with_slash(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                url = factory().supported_interfaces[0].url
                self.assertTrue(url.endswith("/"), f"{name} interface url should end with /")


class TestModes(unittest.TestCase):
    def test_input_modes_json(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                self.assertIn("application/json", list(factory().default_input_modes))

    def test_output_modes_json(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                self.assertIn("application/json", list(factory().default_output_modes))


class TestSkillsPresent(unittest.TestCase):
    def test_orchestrator_skills(self):
        ids = {s.id for s in orchestrator_card().skills}
        self.assertIn("negotiate_slice", ids)
        self.assertIn("arbitrate_escalation", ids)

    def test_ran_skills(self):
        ids = {s.id for s in ran_card().skills}
        self.assertIn("assess_ran", ids)
        self.assertIn("negotiate_ran", ids)

    def test_edge_skills(self):
        ids = {s.id for s in edge_card().skills}
        self.assertIn("assess_edge", ids)
        self.assertIn("negotiate_edge", ids)

    def test_skill_ids_unique_per_card(self):
        for name, factory in ALL_CARDS.items():
            with self.subTest(agent=name):
                ids = [s.id for s in factory().skills]
                self.assertEqual(len(ids), len(set(ids)), f"Duplicate skill IDs in {name}")

    def test_all_skills_have_name_and_description(self):
        for name, factory in ALL_CARDS.items():
            card = factory()
            for skill in card.skills:
                with self.subTest(agent=name, skill=skill.id):
                    self.assertTrue(skill.name, f"{name}.{skill.id} missing name")
                    self.assertTrue(skill.description, f"{name}.{skill.id} missing description")


class TestNoResourceKnobsInSkills(unittest.TestCase):
    """Skills describe WHAT agents do (latency negotiation), never WHAT they have
    (bandwidth MHz, CPU freq, energy watts, cost).  These are internal resource
    knobs that must not be surfaced in the card."""

    def test_no_forbidden_substrings(self):
        for name, factory in ALL_CARDS.items():
            card = factory()
            for skill in card.skills:
                texts = [skill.name.lower(), skill.description.lower()]
                for forbidden in _FORBIDDEN_IN_SKILLS:
                    for text in texts:
                        with self.subTest(agent=name, skill=skill.id, forbidden=forbidden):
                            self.assertNotIn(
                                forbidden, text,
                                f"Resource knob '{forbidden}' found in {name}.{skill.id}"
                            )

    def test_card_description_no_resource_knobs(self):
        for name, factory in ALL_CARDS.items():
            desc = factory().description.lower()
            for forbidden in _FORBIDDEN_IN_SKILLS:
                with self.subTest(agent=name, forbidden=forbidden):
                    self.assertNotIn(forbidden, desc)


if __name__ == "__main__":
    unittest.main()

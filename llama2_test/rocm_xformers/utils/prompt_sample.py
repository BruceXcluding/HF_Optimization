def input_select(seq_len):
    if seq_len  == 1:
# 1
        input_sentences = [
                "A"
                ]
    elif seq_len  == 128:
# 128
        input_sentences = [
                "A semiconductor is a material which is capable of conducting electricity. Semiconductors are used in a wide variety of electronic devices, such as integrated circuits, light emitting diodes, and photovoltaic cells. Semiconductors are also used in a wide variety of other applications, such as in sensors, actuators, and other devices. Semiconductors are typically formed from a semiconductor material, such as silicon, germanium, gallium arsenide, or indium phosphide. Semiconductors are typically formed by growing a thin"
                ]

    elif seq_len  == 1024:
# 1024
        input_sentences = [
                "A semiconductor is a material which has an electrical conductivity value falling between that of a conductor, such as copper, and an insulator, such as glass. Its resistivity falls as its temperature rises; metals behave in the opposite way. Its conducting properties may be altered in useful ways by introducing impurities (doping) into the crystal structure. When two differently doped regions exist in the same crystal, a semiconductor junction is created. The behavior of charge carriers, which include electrons, ions, and electron holes, at these junctions is the basis of diodes, transistors, and most modern electronics. Some examples of semiconductors are silicon, germanium, gallium arsenide, and elements near the so-called metalloid staircase on the periodic table. After silicon, gallium arsenide is the second-most common semiconductor and is used in laser diodes, solar cells, microwave-frequency integrated circuits, and others. Silicon is a critical element for fabricating most electronic circuits. Semiconductor devices can display a range of different useful properties, such as passing current more easily in one direction than the other, showing variable resistance, and having sensitivity to light or heat. Because the electrical properties of a semiconductor material can be modified by doping and by the application of electrical fields or light, devices made from semiconductors can be used for amplification, switching, and energy conversion. The conductivity of silicon is increased by adding a small amount (of the order of 1 in 108) of pentavalent (antimony, phosphorus, or arsenic) or trivalent (boron, gallium, indium) atoms. This process is known as doping, and the resulting semiconductors are known as doped or extrinsic semiconductors. Apart from doping, the conductivity of a semiconductor can be improved by increasing its temperature. This is contrary to the behavior of a metal, in which conductivity decreases with an increase in temperature. The modern understanding of the properties of a semiconductor relies on quantum physics to explain the movement of charge carriers in a crystal lattice.Doping greatly increases the number of charge carriers within the crystal. When a doped semiconductor contains free holes, it is called p-type, and when it contains free electrons, it is known as n-type. The semiconductor materials used in electronic devices are doped under precise conditions to control the concentration and regions of p- and n-type dopants. A single semiconductor device crystal can have many p- and n-type regions; the p–n junctions between these regions are responsible for the useful electronic behavior. Using a hot-point probe, one can determine quickly whether a semiconductor sample is p- or n-type. A few of the properties of semiconductor materials were observed throughout the mid-19th and first decades of the 20th century. The first practical application of semiconductors in electronics was the 1904 development of the cats-whisker detector, a primitive semiconductor diode used in early radio receivers. Developments in quantum physics led in turn to the invention of the transistor in 1947 and the integrated circuit in 1958. Variable electrical conductivity Semiconductors in their natural state are poor conductors because a current requires the flow of electrons, and semiconductors have their valence bands filled, preventing the entire flow of new electrons. Several developed techniques allow semiconducting materials to behave like conducting materials, such as doping or gating. These modifications have two outcomes: n-type and p-type. These refer to the excess or shortage of electrons, respectively. A balanced number of electrons would cause a current to flow throughout the material. Heterojunctions Heterojunctions occur when two differently doped semiconducting materials are joined. For example, a configuration could consist of p-doped and n-doped germanium. This results in an exchange of electrons and holes between the differently doped semiconducting materials. The n-doped germanium would have an excess of electrons, and the p-doped germanium would have an excess of holes. The transfer occurs until an equilibrium is reached by a process called recombination, which causes the migrating electrons from the n-type to come in contact with the migrating holes from the p-type. The result of this process is a narrow strip of immobile ions, which causes an electric field across the junction. Excited electrons A difference in electric potential on a semiconducting material would cause it to leave thermal equilibrium and create a non-equilibrium situation. This introduces electrons and holes to the system, which interact via a process called ambipolar diffusion."
                ]

    elif seq_len  == 2048:
# 2048
        input_sentences = [
                "A semiconductor is a material which has an electrical conductivity value falling between that of a conductor, such as copper, and an insulator, such as glass. Its resistivity falls as its temperature rises; metals behave in the opposite way. Its conducting properties may be altered in useful ways by introducing impurities (doping) into the crystal structure. When two differently doped regions exist in the same crystal, a semiconductor junction is created. The behavior of charge carriers, which include electrons, ions, and electron holes, at these junctions is the basis of diodes, transistors, and most modern electronics. Some examples of semiconductors are silicon, germanium, gallium arsenide, and elements near the so-called metalloid staircase on the periodic table. After silicon, gallium arsenide is the second-most common semiconductor and is used in laser diodes, solar cells, microwave-frequency integrated circuits, and others. Silicon is a critical element for fabricating most electronic circuits. Semiconductor devices can display a range of different useful properties, such as passing current more easily in one direction than the other, showing variable resistance, and having sensitivity to light or heat. Because the electrical properties of a semiconductor material can be modified by doping and by the application of electrical fields or light, devices made from semiconductors can be used for amplification, switching, and energy conversion. The conductivity of silicon is increased by adding a small amount (of the order of 1 in 108) of pentavalent (antimony, phosphorus, or arsenic) or trivalent (boron, gallium, indium) atoms. This process is known as doping, and the resulting semiconductors are known as doped or extrinsic semiconductors. Apart from doping, the conductivity of a semiconductor can be improved by increasing its temperature. This is contrary to the behavior of a metal, in which conductivity decreases with an increase in temperature. The modern understanding of the properties of a semiconductor relies on quantum physics to explain the movement of charge carriers in a crystal lattice.Doping greatly increases the number of charge carriers within the crystal. When a doped semiconductor contains free holes, it is called p-type, and when it contains free electrons, it is known as n-type. The semiconductor materials used in electronic devices are doped under precise conditions to control the concentration and regions of p- and n-type dopants. A single semiconductor device crystal can have many p- and n-type regions; the p–n junctions between these regions are responsible for the useful electronic behavior. Using a hot-point probe, one can determine quickly whether a semiconductor sample is p- or n-type. A few of the properties of semiconductor materials were observed throughout the mid-19th and first decades of the 20th century. The first practical application of semiconductors in electronics was the 1904 development of the cats-whisker detector, a primitive semiconductor diode used in early radio receivers. Developments in quantum physics led in turn to the invention of the transistor in 1947 and the integrated circuit in 1958. Variable electrical conductivity Semiconductors in their natural state are poor conductors because a current requires the flow of electrons, and semiconductors have their valence bands filled, preventing the entire flow of new electrons. Several developed techniques allow semiconducting materials to behave like conducting materials, such as doping or gating. These modifications have two outcomes: n-type and p-type. These refer to the excess or shortage of electrons, respectively. A balanced number of electrons would cause a current to flow throughout the material. Heterojunctions Heterojunctions occur when two differently doped semiconducting materials are joined. For example, a configuration could consist of p-doped and n-doped germanium. This results in an exchange of electrons and holes between the differently doped semiconducting materials. The n-doped germanium would have an excess of electrons, and the p-doped germanium would have an excess of holes. The transfer occurs until an equilibrium is reached by a process called recombination, which causes the migrating electrons from the n-type to come in contact with the migrating holes from the p-type. The result of this process is a narrow strip of immobile ions, which causes an electric field across the junction. Excited electrons A difference in electric potential on a semiconducting material would cause it to leave thermal equilibrium and create a non-equilibrium situation. This introduces electrons and holes to the system, which interact via a process called ambipolar diffusion. Whenever thermal equilibrium is disturbed in a semiconducting material, the number of holes and electrons changes. Such disruptions can occur as a result of a temperature difference or photons, which can enter the system and create electrons and holes. The processes that create or annihilate electrons and holes are called generation and recombination, respectively. Light emission In certain semiconductors, excited electrons can relax by emitting light instead of producing heat. These semiconductors are used in the construction of light-emitting diodes and fluorescent quantum dots. A semiconductor is a material which has an electrical conductivity value falling between that of a conductor, such as copper, and an insulator, such as glass. Its resistivity falls as its temperature rises; metals behave in the opposite way. Its conducting properties may be altered in useful ways by introducing impurities (doping) into the crystal structure. When two differently doped regions exist in the same crystal, a semiconductor junction is created. The behavior of charge carriers, which include electrons, ions, and electron holes, at these junctions is the basis of diodes, transistors, and most modern electronics. Some examples of semiconductors are silicon, germanium, gallium arsenide, and elements near the so-called metalloid staircase on the periodic table. After silicon, gallium arsenide is the second-most common semiconductor and is used in laser diodes, solar cells, microwave-frequency integrated circuits, and others. Silicon is a critical element for fabricating most electronic circuits. Semiconductor devices can display a range of different useful properties, such as passing current more easily in one direction than the other, showing variable resistance, and having sensitivity to light or heat. Because the electrical properties of a semiconductor material can be modified by doping and by the application of electrical fields or light, devices made from semiconductors can be used for amplification, switching, and energy conversion. The conductivity of silicon is increased by adding a small amount (of the order of 1 in 108) of pentavalent (antimony, phosphorus, or arsenic) or trivalent (boron, gallium, indium) atoms. This process is known as doping, and the resulting semiconductors are known as doped or extrinsic semiconductors. Apart from doping, the conductivity of a semiconductor can be improved by increasing its temperature. This is contrary to the behavior of a metal, in which conductivity decreases with an increase in temperature. The modern understanding of the properties of a semiconductor relies on quantum physics to explain the movement of charge carriers in a crystal lattice.Doping greatly increases the number of charge carriers within the crystal. When a doped semiconductor contains free holes, it is called p-type, and when it contains free electrons, it is known as n-type. The semiconductor materials used in electronic devices are doped under precise conditions to control the concentration and regions of p- and n-type dopants. A single semiconductor device crystal can have many p- and n-type regions; the p–n junctions between these regions are responsible for the useful electronic behavior. Using a hot-point probe, one can determine quickly whether a semiconductor sample is p- or n-type. A few of the properties of semiconductor materials were observed throughout the mid-19th and first decades of the 20th century. The first practical application of semiconductors in electronics was the 1904 development of the cats-whisker detector, a primitive semiconductor diode used in early radio receivers. Developments in quantum physics led in turn to the invention of the transistor in 1947 and the integrated circuit in 1958. Variable electrical conductivity Semiconductors in their natural state are poor conductors because a current requires the flow of electrons, and semiconductors have their valence bands filled, preventing the entire flow of new electrons. Several developed techniques allow semiconducting materials to behave like conducting materials, such as doping or gating. These modifications have two outcomes: n"
                ]

    elif seq_len  == 4096:
# 4096
        input_sentences = [
                "A semiconductor is a material which has an electrical conductivity value falling between that of a conductor, such as copper, and an insulator, such as glass. Its resistivity falls as its temperature rises; metals behave in the opposite way. Its conducting properties may be altered in useful ways by introducing impurities (doping) into the crystal structure. When two differently doped regions exist in the same crystal, a semiconductor junction is created. The behavior of charge carriers, which include electrons, ions, and electron holes, at these junctions is the basis of diodes, transistors, and most modern electronics. Some examples of semiconductors are silicon, germanium, gallium arsenide, and elements near the so-called metalloid staircase on the periodic table. After silicon, gallium arsenide is the second-most common semiconductor and is used in laser diodes, solar cells, microwave-frequency integrated circuits, and others. Silicon is a critical element for fabricating most electronic circuits. Semiconductor devices can display a range of different useful properties, such as passing current more easily in one direction than the other, showing variable resistance, and having sensitivity to light or heat. Because the electrical properties of a semiconductor material can be modified by doping and by the application of electrical fields or light, devices made from semiconductors can be used for amplification, switching, and energy conversion. The conductivity of silicon is increased by adding a small amount (of the order of 1 in 108) of pentavalent (antimony, phosphorus, or arsenic) or trivalent (boron, gallium, indium) atoms. This process is known as doping, and the resulting semiconductors are known as doped or extrinsic semiconductors. Apart from doping, the conductivity of a semiconductor can be improved by increasing its temperature. This is contrary to the behavior of a metal, in which conductivity decreases with an increase in temperature. The modern understanding of the properties of a semiconductor relies on quantum physics to explain the movement of charge carriers in a crystal lattice.Doping greatly increases the number of charge carriers within the crystal. When a doped semiconductor contains free holes, it is called p-type, and when it contains free electrons, it is known as n-type. The semiconductor materials used in electronic devices are doped under precise conditions to control the concentration and regions of p- and n-type dopants. A single semiconductor device crystal can have many p- and n-type regions; the p–n junctions between these regions are responsible for the useful electronic behavior. Using a hot-point probe, one can determine quickly whether a semiconductor sample is p- or n-type. A few of the properties of semiconductor materials were observed throughout the mid-19th and first decades of the 20th century. The first practical application of semiconductors in electronics was the 1904 development of the cats-whisker detector, a primitive semiconductor diode used in early radio receivers. Developments in quantum physics led in turn to the invention of the transistor in 1947 and the integrated circuit in 1958. Variable electrical conductivity Semiconductors in their natural state are poor conductors because a current requires the flow of electrons, and semiconductors have their valence bands filled, preventing the entire flow of new electrons. Several developed techniques allow semiconducting materials to behave like conducting materials, such as doping or gating. These modifications have two outcomes: n-type and p-type. These refer to the excess or shortage of electrons, respectively. A balanced number of electrons would cause a current to flow throughout the material. Heterojunctions Heterojunctions occur when two differently doped semiconducting materials are joined. For example, a configuration could consist of p-doped and n-doped germanium. This results in an exchange of electrons and holes between the differently doped semiconducting materials. The n-doped germanium would have an excess of electrons, and the p-doped germanium would have an excess of holes. The transfer occurs until an equilibrium is reached by a process called recombination, which causes the migrating electrons from the n-type to come in contact with the migrating holes from the p-type. The result of this process is a narrow strip of immobile ions, which causes an electric field across the junction. Excited electrons A difference in electric potential on a semiconducting material would cause it to leave thermal equilibrium and create a non-equilibrium situation. This introduces electrons and holes to the system, which interact via a process called ambipolar diffusion. Whenever thermal equilibrium is disturbed in a semiconducting material, the number of holes and electrons changes. Such disruptions can occur as a result of a temperature difference or photons, which can enter the system and create electrons and holes. The processes that create or annihilate electrons and holes are called generation and recombination, respectively. Light emission In certain semiconductors, excited electrons can relax by emitting light instead of producing heat. These semiconductors are used in the construction of light-emitting diodes and fluorescent quantum dots. A semiconductor is a material which has an electrical conductivity value falling between that of a conductor, such as copper, and an insulator, such as glass. Its resistivity falls as its temperature rises; metals behave in the opposite way. Its conducting properties may be altered in useful ways by introducing impurities (doping) into the crystal structure. When two differently doped regions exist in the same crystal, a semiconductor junction is created. The behavior of charge carriers, which include electrons, ions, and electron holes, at these junctions is the basis of diodes, transistors, and most modern electronics. Some examples of semiconductors are silicon, germanium, gallium arsenide, and elements near the so-called metalloid staircase on the periodic table. After silicon, gallium arsenide is the second-most common semiconductor and is used in laser diodes, solar cells, microwave-frequency integrated circuits, and others. Silicon is a critical element for fabricating most electronic circuits. Semiconductor devices can display a range of different useful properties, such as passing current more easily in one direction than the other, showing variable resistance, and having sensitivity to light or heat. Because the electrical properties of a semiconductor material can be modified by doping and by the application of electrical fields or light, devices made from semiconductors can be used for amplification, switching, and energy conversion. The conductivity of silicon is increased by adding a small amount (of the order of 1 in 108) of pentavalent (antimony, phosphorus, or arsenic) or trivalent (boron, gallium, indium) atoms. This process is known as doping, and the resulting semiconductors are known as doped or extrinsic semiconductors. Apart from doping, the conductivity of a semiconductor can be improved by increasing its temperature. This is contrary to the behavior of a metal, in which conductivity decreases with an increase in temperature. The modern understanding of the properties of a semiconductor relies on quantum physics to explain the movement of charge carriers in a crystal lattice.Doping greatly increases the number of charge carriers within the crystal. When a doped semiconductor contains free holes, it is called p-type, and when it contains free electrons, it is known as n-type. The semiconductor materials used in electronic devices are doped under precise conditions to control the concentration and regions of p- and n-type dopants. A single semiconductor device crystal can have many p- and n-type regions; the p–n junctions between these regions are responsible for the useful electronic behavior. Using a hot-point probe, one can determine quickly whether a semiconductor sample is p- or n-type. A few of the properties of semiconductor materials were observed throughout the mid-19th and first decades of the 20th century. The first practical application of semiconductors in electronics was the 1904 development of the cats-whisker detector, a primitive semiconductor diode used in early radio receivers. Developments in quantum physics led in turn to the invention of the transistor in 1947 and the integrated circuit in 1958. Variable electrical conductivity Semiconductors in their natural state are poor conductors because a current requires the flow of electrons, and semiconductors have their valence bands filled, preventing the entire flow of new electrons. Several developed techniques allow semiconducting materials to behave like conducting materials, such as doping or gating. These modifications have two outcomes: n. A semiconductor is a material which has an electrical conductivity value falling between that of a conductor, such as copper, and an insulator, such as glass. Its resistivity falls as its temperature rises; metals behave in the opposite way. Its conducting properties may be altered in useful ways by introducing impurities (doping) into the crystal structure. When two differently doped regions exist in the same crystal, a semiconductor junction is created. The behavior of charge carriers, which include electrons, ions, and electron holes, at these junctions is the basis of diodes, transistors, and most modern electronics. Some examples of semiconductors are silicon, germanium, gallium arsenide, and elements near the so-called metalloid staircase on the periodic table. After silicon, gallium arsenide is the second-most common semiconductor and is used in laser diodes, solar cells, microwave-frequency integrated circuits, and others. Silicon is a critical element for fabricating most electronic circuits. Semiconductor devices can display a range of different useful properties, such as passing current more easily in one direction than the other, showing variable resistance, and having sensitivity to light or heat. Because the electrical properties of a semiconductor material can be modified by doping and by the application of electrical fields or light, devices made from semiconductors can be used for amplification, switching, and energy conversion. The conductivity of silicon is increased by adding a small amount (of the order of 1 in 108) of pentavalent (antimony, phosphorus, or arsenic) or trivalent (boron, gallium, indium) atoms. This process is known as doping, and the resulting semiconductors are known as doped or extrinsic semiconductors. Apart from doping, the conductivity of a semiconductor can be improved by increasing its temperature. This is contrary to the behavior of a metal, in which conductivity decreases with an increase in temperature. The modern understanding of the properties of a semiconductor relies on quantum physics to explain the movement of charge carriers in a crystal lattice.Doping greatly increases the number of charge carriers within the crystal. When a doped semiconductor contains free holes, it is called p-type, and when it contains free electrons, it is known as n-type. The semiconductor materials used in electronic devices are doped under precise conditions to control the concentration and regions of p- and n-type dopants. A single semiconductor device crystal can have many p- and n-type regions; the p–n junctions between these regions are responsible for the useful electronic behavior. Using a hot-point probe, one can determine quickly whether a semiconductor sample is p- or n-type. A few of the properties of semiconductor materials were observed throughout the mid-19th and first decades of the 20th century. The first practical application of semiconductors in electronics was the 1904 development of the cats-whisker detector, a primitive semiconductor diode used in early radio receivers. Developments in quantum physics led in turn to the invention of the transistor in 1947 and the integrated circuit in 1958. Variable electrical conductivity Semiconductors in their natural state are poor conductors because a current requires the flow of electrons, and semiconductors have their valence bands filled, preventing the entire flow of new electrons. Several developed techniques allow semiconducting materials to behave like conducting materials, such as doping or gating. These modifications have two outcomes: n-type and p-type. These refer to the excess or shortage of electrons, respectively. A balanced number of electrons would cause a current to flow throughout the material. Heterojunctions Heterojunctions occur when two differently doped semiconducting materials are joined. For example, a configuration could consist of p-doped and n-doped germanium. This results in an exchange of electrons and holes between the differently doped semiconducting materials. The n-doped germanium would have an excess of electrons, and the p-doped germanium would have an excess of holes. The transfer occurs until an equilibrium is reached by a process called recombination, which causes the migrating electrons from the n-type to come in contact with the migrating holes from the p-type. The result of this process is a narrow strip of immobile ions, which causes an electric field across the junction. Excited electrons A difference in electric potential on a semiconducting material would cause it to leave thermal equilibrium and create a non-equilibrium situation. This introduces electrons and holes to the system, which interact via a process called ambipolar diffusion. Whenever thermal equilibrium is disturbed in a semiconducting material, the number of holes and electrons changes. Such disruptions can occur as a result of a temperature difference or photons, which can enter the system and create electrons and holes. The processes that create or annihilate electrons and holes are called generation and recombination, respectively. Light emission In certain semiconductors, excited electrons can relax by emitting light instead of producing heat. These semiconductors are used in the construction of light-emitting diodes and fluorescent quantum dots. A semiconductor is a material which has an electrical conductivity value falling between that of a conductor, such as copper, and an insulator, such as glass. Its resistivity falls as its temperature rises; metals behave in the opposite way. Its conducting properties may be altered in useful ways by introducing impurities (doping) into the crystal structure. When two differently doped regions exist in the same crystal, a semiconductor junction is created. The behavior of charge carriers, which include electrons, ions, and electron holes, at these junctions is the basis of diodes, transistors, and most modern electronics. Some examples of semiconductors are silicon, germanium, gallium arsenide, and elements near the so-called metalloid staircase on the periodic table. After silicon, gallium arsenide is the second-most common semiconductor and is used in laser diodes, solar cells, microwave-frequency integrated circuits, and others. Silicon is a critical element for fabricating most electronic circuits. Semiconductor devices can display a range of different useful properties, such as passing current more easily in one direction than the other, showing variable resistance, and having sensitivity to light or heat. Because the electrical properties of a semiconductor material can be modified by doping and by the application of electrical fields or light, devices made from semiconductors can be used for amplification, switching, and energy conversion. The conductivity of silicon is increased by adding a small amount (of the order of 1 in 108) of pentavalent (antimony, phosphorus, or arsenic) or trivalent (boron, gallium, indium) atoms. This process is known as doping, and the resulting semiconductors are known as doped or extrinsic semiconductors. Apart from doping, the conductivity of a semiconductor can be improved by increasing its temperature. This is contrary to the behavior of a metal, in which conductivity decreases with an increase in temperature. The modern understanding of the properties of a semiconductor relies on quantum physics to explain the movement of charge carriers in a crystal lattice.Doping greatly increases the number of charge carriers within the crystal. When a doped semiconductor contains free holes, it is called p-type, and when it contains free electrons, it is known as n-type. The semiconductor materials used in electronic devices are doped under precise conditions to control the concentration and regions of p- and n-type dopants. A single semiconductor device crystal can have many p- and n-type regions; the p–n junctions between these regions are responsible for the useful electronic behavior. Using a hot-point probe, one can determine quickly whether a semiconductor sample is p- or n-type. A few of the properties of semiconductor materials were observed throughout the mid-19th and first decades of the 20th century. The first practical application of semiconductors in electronics was the 1904 development of the cats-whisker detector, a primitive semiconductor diode used in early radio receivers. Developments in quantum physics led in turn to the invention of the transistor in 1947 and the integrated circuit in 1958. Variable electrical conductivity Semiconductors in their natural state are poor conductors because a current requires the flow of electrons, and semiconductors have their valence bands filled, preventing the entire flow of new electrons. Several developed techniques allow semiconducting materials to behave like conducting materials, such as doping or gating. These modifications have two outcomes: n"]
    return input_sentences


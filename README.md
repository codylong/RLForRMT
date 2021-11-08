# RLForRMT

The motivation for this project is to understand whether machine learning, in particular 
reinforcement learning, can help to derive useful random matrix approximations to string theory data. 
This project uses [OpenAI Gym](https://gym.openai.com/) and [ChainerRL](https://github.com/chainer/chainerrl)
for the RL architectures and environments. 

The data of interest is numerical metrics on the Kähler moduli space of Calabi-Yau threefold
hypersurfaces in weak Fano toric fourfolds. To utilize RL, we use a [Bergman approximation](https://en.wikipedia.org/wiki/Bergman_metric) to 
the metric on moduli space, where the coefficients for each line bundle section are treated
as free parameters, to be optimized by RL to match the Calabi-Yau data. 


The data of interest is obtained via triangulations of 4d reflexive polytopes in the 
[Kreuzer-Skarke database](http://hep.itp.tuwien.ac.at/~kreuzer/CY/), as well as
a python-to-geometry pipeline that can be found in Cody Long's github. 

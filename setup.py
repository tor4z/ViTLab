from distutils.core import setup
import datetime


def gen_code():
    d = datetime.datetime.now()
    date_str = d.strftime('%Y%m%d%H%M%S')
    
    return f'dev{date_str}'


__version__ = f'0.0.1.{gen_code()}'


setup(name='vitlab',
      version=__version__,
      description='Vit Lab',
      author='tor4z',
      author_email='vwenjie@hotmail.com',
      install_requires=[
            'torch',
            'numpy',
            # 'timm',
            # 'ml-collections',
            'jupyterlab',
            'matplotlib',
      ],
      packages=['vitlab',
                'vitlab.dataset',
                'vitlab.network',
                'vitlab.network.vit',
                'vitlab.network.resnet',
                'vitlab.trainer',
                'vitlab.trainer.vit',
                'vitlab.trainer.resnet',
                'vitlab.utils',
      ]
)

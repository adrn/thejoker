***********
What's new?
***********

v1.0
====

TODO: API change! and new features...

v0.3
====

* **thejoker now has a new sampling mode: TheJoker.iterative_rejection_sample**.
  The default ``rejection_sample()`` method always uses the full prior samples
  library passed to it and will be inefficient when the data are noisy or
  uninformative. The new ``iterative_rejection_sample()`` method instead accepts
  a requested number of samples, and will try to predict how many prior samples
  it needs to use in order to return the desired number of posterior samples.

* **thejoker now supports sampling over long-term velocity trends**. The
  default assumption for this sampler is that all systems are two-body systems.
  However, the sampler now supports hierarchical triple systems in some special
  cases. When the orbit of the outer body in a triple system is much longer than
  the orbit of the inner binary, and the outer orbital period is not fully
  observed, the influence of the outer body can be described by a long-term
  trend in the radial velocity that is approximately polynomial in time.
  ``thejoker`` now supports sampling over polynomial velocity trends with a
  specified polynomial degree. The default still assumes a constant velocity
  offset, i.e. that the barycentric velocity of the binary is constant.

* **The sampler is ~4–10 times faster**. Thanks to some improvements on the
  implementation of the sampler (now in C + Cython), the sampler is
  significantly faster than the initial release.

* **The sampler now optionally returns the prior and likelihood values of all
  posterior samples**. By specifying ``return_logprobs=True`` when running
  ``rejection_sample()`` or ``iterative_rejection_sample()``, the sampler now
  returns log-prior and log-likelihood values for any returned samples.

/*
 * Copyright (c) 2013 L2jMobius
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
 * IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.l2jmobius.gameserver.managers;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import org.l2jmobius.commons.threads.ThreadPool;
import org.l2jmobius.commons.util.Rnd;
import org.l2jmobius.gameserver.config.PlayerConfig;
import org.l2jmobius.gameserver.config.custom.RLAgentEnvironmentConfig;
import org.l2jmobius.gameserver.config.custom.RLAgentEnvironmentConfig.BuffEntry;
import org.l2jmobius.gameserver.data.xml.SkillData;
import org.l2jmobius.gameserver.model.World;
import org.l2jmobius.gameserver.model.actor.Player;
import org.l2jmobius.gameserver.model.skill.AbnormalType;
import org.l2jmobius.gameserver.model.skill.Skill;

/**
 * Buffs of the RL agents: every life an agent gets one buff set, picked at random with the {@code RLBuffSetWeights}.
 * <ul>
 * <li>{@code none}: no buffs;</li>
 * <li>{@code newbie}: the Newbie Helper buffs, with their level limits (an agent outside them falls back to another set);</li>
 * <li>{@code support}: the buffs of one random support class ({@code RLBuffSupport.<Class>});</li>
 * <li>{@code dances_songs}: Swordsinger songs and Bladedancer dances;</li>
 * <li>{@code full}: a full NPC buffer set.</li>
 * </ul>
 * A buff is given at the highest level whose magic level the agent has reached, as if a support of the same level cast it; a buff whose first level is above the agent is left out, and so is a buff the agent casts itself (the same effect as one of its own active buffs: a Bladedancer gets no dances, a Prophet no Might). One buff per abnormal type is kept and the set is cut to the buff and dance slots, so the periodic refresh of expired buffs never pushes another one of the set out.
 */
final class RLBuffs
{
	private static final Logger LOGGER = Logger.getLogger(RLBuffs.class.getName());
	
	/** Agent object id -> the buffs of its current life. */
	private final Map<Integer, List<Skill>> _assigned = new ConcurrentHashMap<>();
	
	RLBuffs()
	{
		if (RLAgentEnvironmentConfig.RL_BUFF_REFRESH_INTERVAL > 0)
		{
			ThreadPool.scheduleAtFixedRate(this::refreshAll, RLAgentEnvironmentConfig.RL_BUFF_REFRESH_INTERVAL, RLAgentEnvironmentConfig.RL_BUFF_REFRESH_INTERVAL);
		}
	}
	
	/**
	 * Picks the buff set of a new life and casts it.
	 * @param player the agent
	 * @return the name of the picked set
	 */
	String assign(Player player)
	{
		String set = pickSet();
		List<Skill> buffs = build(player, set);
		if (buffs.isEmpty() && !set.equals("none"))
		{
			// e.g. the newbie buffs outside levels 8-25: take any set that gives something.
			for (String other : RLAgentEnvironmentConfig.RL_BUFF_SET_WEIGHTS.keySet())
			{
				if (!other.equals("none") && !other.equals("newbie"))
				{
					set = other;
					buffs = build(player, set);
					break;
				}
			}
		}
		
		if (buffs.isEmpty())
		{
			_assigned.remove(player.getObjectId());
			return "none";
		}
		_assigned.put(player.getObjectId(), buffs);
		refresh(player, buffs);
		return set;
	}
	
	/** Re-levels the buffs of the current set after a level-up. */
	void relevel(Player player)
	{
		final List<Skill> current = _assigned.get(player.getObjectId());
		if (current != null)
		{
			final List<Skill> buffs = new ArrayList<>();
			for (Skill skill : current)
			{
				final int skillLevel = levelFor(skill.getId(), player.getLevel());
				final Skill better = skillLevel > skill.getLevel() ? SkillData.getInstance().getSkill(skill.getId(), skillLevel) : null;
				buffs.add(better != null ? better : skill);
			}
			_assigned.put(player.getObjectId(), buffs);
		}
	}
	
	void release(Player player)
	{
		_assigned.remove(player.getObjectId());
	}
	
	private void refreshAll()
	{
		for (Map.Entry<Integer, List<Skill>> entry : _assigned.entrySet())
		{
			final Player player = World.getInstance().getPlayer(entry.getKey());
			if (player == null)
			{
				_assigned.remove(entry.getKey());
				continue;
			}
			try
			{
				refresh(player, entry.getValue());
			}
			catch (Exception e)
			{
				LOGGER.warning(RLBuffs.class.getSimpleName() + ": buff refresh of " + player.getName() + " failed: " + e.getMessage());
			}
		}
	}
	
	private static void refresh(Player player, List<Skill> buffs)
	{
		if (!player.isOnline() || player.isDead())
		{
			return;
		}
		for (Skill skill : buffs)
		{
			if (!player.isAffectedBySkill(skill.getId()))
			{
				skill.applyEffects(player, player);
			}
		}
	}
	
	private static String pickSet()
	{
		int total = 0;
		for (int weight : RLAgentEnvironmentConfig.RL_BUFF_SET_WEIGHTS.values())
		{
			total += weight;
		}
		if (total <= 0)
		{
			return "none";
		}
		
		int pick = Rnd.get(total);
		for (Map.Entry<String, Integer> entry : RLAgentEnvironmentConfig.RL_BUFF_SET_WEIGHTS.entrySet())
		{
			pick -= entry.getValue();
			if (pick < 0)
			{
				return entry.getKey();
			}
		}
		return "none";
	}
	
	private static List<Skill> build(Player player, String set)
	{
		final List<BuffEntry> entries = switch (set)
		{
			case "newbie" -> RLAgentEnvironmentConfig.RL_BUFF_NEWBIE;
			case "support" ->
			{
				final List<List<BuffEntry>> classes = new ArrayList<>(RLAgentEnvironmentConfig.RL_BUFF_SUPPORT.values());
				yield classes.isEmpty() ? List.of() : classes.get(Rnd.get(classes.size()));
			}
			case "dances_songs" -> RLAgentEnvironmentConfig.RL_BUFF_DANCES_SONGS;
			case "full" -> RLAgentEnvironmentConfig.RL_BUFF_FULL;
			default -> List.of();
		};
		
		final int level = player.getLevel();
		final List<Skill> result = new ArrayList<>();
		// Effects the agent has its own buffs for count as taken: it casts them itself.
		final Set<AbnormalType> types = new HashSet<>(ownBuffTypes(player));
		int buffSlots = player.getStat().getMaxBuffCount();
		int danceSlots = PlayerConfig.DANCES_MAX_AMOUNT;
		for (BuffEntry entry : entries)
		{
			if ((level < entry.minLevel()) || (level > entry.maxLevel()) || (entry.mageOnly() && !RLAgentManager.isMageBuild(player)) || (entry.fighterOnly() && RLAgentManager.isMageBuild(player)))
			{
				continue;
			}
			
			final int skillLevel = levelFor(entry.skillId(), level);
			final Skill skill = skillLevel > 0 ? SkillData.getInstance().getSkill(entry.skillId(), skillLevel) : null;
			if ((skill == null) || !types.add(skill.getAbnormalType()))
			{
				continue; // unknown, above the agent's level, or the same effect as an earlier buff (Might after Greater Might)
			}
			if (skill.isDance())
			{
				if (danceSlots-- <= 0)
				{
					continue;
				}
			}
			else if (buffSlots-- <= 0)
			{
				continue;
			}
			result.add(skill);
		}
		return result;
	}
	
	/** Effect types of the agent's own active buffs (continuous, not debuffs, not toggles). */
	private static Set<AbnormalType> ownBuffTypes(Player player)
	{
		final Set<AbnormalType> types = new HashSet<>();
		for (Skill skill : player.getSkills().values())
		{
			if (!skill.isPassive() && !skill.isToggle() && skill.isContinuous() && !skill.isDebuff() && (skill.getAbnormalType() != AbnormalType.NONE))
			{
				types.add(skill.getAbnormalType());
			}
		}
		return types;
	}
	
	/** Highest level of the skill whose magic level does not exceed the agent's level, 0 when none does (or the skill is unknown). */
	private static int levelFor(int skillId, int playerLevel)
	{
		final int maxLevel = SkillData.getInstance().getMaxLevel(skillId);
		int best = 0;
		for (int level = 1; level <= maxLevel; level++)
		{
			final Skill skill = SkillData.getInstance().getSkill(skillId, level);
			if ((skill != null) && (skill.getMagicLevel() <= playerLevel))
			{
				best = level;
			}
		}
		return best;
	}
}
